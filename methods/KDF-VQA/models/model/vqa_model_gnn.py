import numpy as np
from einops import rearrange
from typing import Tuple, Union, Optional
from dataclasses import dataclass, field

import torch
import transformers
import torch.nn.functional as F
import torchvision.models as models
# from torchvision.models.models import resnet50, ResNet50_Weights

from torch import nn
# from torch.nn import TransformerEncoder
from transformers import AutoModel, BertConfig, AutoTokenizer, CLIPVisionModel, CLIPVisionConfig, LlamaTokenizer
from models.llama.blocks import Transformer
from models.pmc_oa.blocks import Transformer, AttentionPool2d
from models.pmc_oa.pmc_clip import PMC_CLIP
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from utils.layers import *
from torch.autograd import Variable
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter
from torch_geometric.nn.inits import glorot, zeros

# from blocks import Transformer

# from peft import (
#     get_peft_model,
#     LoraConfig,
#     PrefixTuningConfig,
#     PromptEncoderConfig,
#     PromptTuningConfig,
#     TaskType,
# )

@dataclass
class CLIPTextCfg:
    bert_model_name: str = 'base'
    context_length: int = 77
    # vocab_size: int = 32000
    vocab_size: int = 30522
    # vocab_size: int = 28895
    width: int = 768
    heads: int = 8
    layers: int = 12
    fusion_layers: int = 1  # layers of fusion_module
    MOMENTUM: float = 0.5  # 0.99

    # bert_model_name: str = 'base'
    # context_length: int = 77
    # vocab_size: int = 30522
    # width: int = 512
    # heads: int = 8
    # layers: int = 12
    # fusion_layers: int = 1  # layers of fusion_module
    # MOMENTUM: float = 0.5  # 0.99


@dataclass
class PEFTArguments:
    peft_mode: str = field(default="lora")
    lora_rank: int = field(default=8)
    num_virtual_tokens: int = field(default=32)  # Used for prompt tuning, prefix tuning and p-tuning
    mapping_hidden_dim: int = field(default=1024)

def make_one_hot(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        (N, ), where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.
    Returns : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    '''
    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), C).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target

class GATConvE(MessagePassing):
    """
    Args:
        emb_dim (int): dimensionality of GNN hidden states
        n_ntype (int): number of node types (e.g. 4)
        n_etype (int): number of edge relation types (e.g. 38)
    """
    def __init__(self, args, emb_dim, n_ntype, n_etype, edge_encoder, head_count=4, aggr="add"):
        super(GATConvE, self).__init__(aggr=aggr)
        self.args = args

        # print("emb_dim:", emb_dim)
        # print("head_count:", head_count)
        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim

        self.n_ntype = n_ntype; self.n_etype = n_etype
        self.edge_encoder = edge_encoder

        #For attention
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        self.linear_key = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(2*emb_dim, head_count * self.dim_per_head)

        self._alpha = None

        #For final MLP
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))


    def forward(self, x, edge_index, edge_type, node_type, node_feature_extra, return_attention_weights=False):
        # x: [N, emb_dim]
        # edge_index: [2, E]
        # edge_type [E,] -> edge_attr: [E, 39] / self_edge_attr: [N, 39]
        # node_type [N,] -> headtail_attr [E, 8(=4+4)] / self_headtail_attr: [N, 8]
        # node_feature_extra [N, dim]

        #Prepare edge feature
        edge_vec = make_one_hot(edge_type, self.n_etype +1) #[E, 39]
        self_edge_vec = torch.zeros(x.size(0), self.n_etype +1).to(edge_vec.device)
        self_edge_vec[:,self.n_etype] = 1

        # print(x.size())
        # print(self_edge_vec.size())  # E=664
        # print(edge_index.size()) # E=664
        # print(edge_type.size()) # E=664
        # print(node_type.size()) # N=100
        # print(edge_vec.size()) # [664, 27]

        head_type = node_type[edge_index[0]] #[E,] #head=src
        tail_type = node_type[edge_index[1]] #[E,] #tail=tgt
        head_vec = make_one_hot(head_type, self.n_ntype) #[E,4]
        tail_vec = make_one_hot(tail_type, self.n_ntype) #[E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1) #[E,8]
        self_head_vec = make_one_hot(node_type, self.n_ntype) #[N,4]
        self_headtail_vec = torch.cat([self_head_vec, self_head_vec], dim=1) #[N,8]

        # print(head_type.size())  # E=664
        # print(tail_type.size())  # E=664
        # print(head_vec.size())  # [664, 4]
        # print(tail_vec.size())  # [664, 4]
        # print(headtail_vec.size())  # [664, 8]
        # print(self_head_vec.size())  # [100, 4]
        # print(self_headtail_vec.size())  # [100, 8]

        edge_vec = torch.cat([edge_vec, self_edge_vec], dim=0) #[E+N, ?]
        headtail_vec = torch.cat([headtail_vec, self_headtail_vec], dim=0) #[E+N, ?]

        # print(edge_vec.size()) # [764, 27]
        # print(headtail_vec.size()) # [764, 8]

        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1)) #[E+N, emb_dim]

        #Add self loops to edge_index
        loop_index = torch.arange(0, x.size(0), dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)  #[2, E+N]

        x = torch.cat([x, node_feature_extra], dim=1)
        x = (x, x)
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings) #[N, emb_dim]
        out = self.mlp(aggr_out)

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out


    def message(self, edge_index, x_i, x_j, edge_attr): #i: tgt, j:src
        # print ("edge_attr.size()", edge_attr.size()) #[E, emb_dim]
        # print ("x_j.size()", x_j.size()) #[E, emb_dim]
        # print ("x_i.size()", x_i.size()) #[E, emb_dim]
        assert len(edge_attr.size()) == 2
        assert edge_attr.size(1) == self.emb_dim
        assert x_i.size(1) == x_j.size(1) == 2*self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(0) == edge_index.size(1)

        key   = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head) #[E, heads, _dim]
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head) #[E, heads, _dim]
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head) #[E, heads, _dim]


        query = query / math.sqrt(self.dim_per_head)
        scores = (query * key).sum(dim=2) #[E, heads]
        src_node_index = edge_index[0] #[E,]
        alpha = softmax(scores, src_node_index) #[E, heads] #group by src side node
        self._alpha = alpha

        #adjust by outgoing degree of src
        E = edge_index.size(1)            #n_edges
        N = int(src_node_index.max()) + 1 #n_nodes
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_edge_count = scatter(ones, src_node_index, dim=0, dim_size=N, reduce='sum')[src_node_index] #[E,]
        assert len(src_node_edge_count.size()) == 1 and len(src_node_edge_count) == E
        alpha = alpha * src_node_edge_count.unsqueeze(1) #[E, heads]

        out = msg * alpha.view(-1, self.head_count, 1) #[E, heads, _dim]
        return out.view(-1, self.head_count * self.dim_per_head)  #[E, emb_dim]


class GNN_Message_Passing(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, input_size, hidden_size, output_size,
                    dropout=0.1):
        super().__init__()
        assert input_size == output_size
        self.args = args
        self.n_ntype = n_ntype
        self.n_etype = n_etype

        assert input_size == hidden_size
        self.hidden_size = hidden_size

        self.emb_node_type = nn.Linear(self.n_ntype, hidden_size//2)

        self.basis_f = 'sin' #['id', 'linact', 'sin', 'none']
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, hidden_size//2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, hidden_size//2)
            self.emb_score = nn.Linear(hidden_size//2, hidden_size//2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(hidden_size//2, hidden_size//2)

        self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(n_etype +1 + n_ntype *2, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size))


        self.k = k
        self.gnn_layers = nn.ModuleList([GATConvE(args, hidden_size, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])


        self.Vh = nn.Linear(input_size, output_size)
        self.Vx = nn.Linear(hidden_size, output_size)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout


    def mp_helper(self, _X, edge_index, edge_type, _node_type, _node_feature_extra):
        for _ in range(self.k):
            _X = self.gnn_layers[_](_X, edge_index, edge_type, _node_type, _node_feature_extra)
            _X = self.activation(_X)
            _X = F.dropout(_X, self.dropout_rate, training = self.training)
        return _X


    def forward(self, H, A, node_type, node_score, cache_output=False):
        """
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type)
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        """
        _batch_size, _n_nodes = node_type.size()
        # _n_nodes = 100 = n_node
        #Embed type
        T = make_one_hot(node_type.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T)) #[batch_size, n_node, dim/2]

        #Embed score
        if self.basis_f == 'sin':
            js = torch.arange(self.hidden_size//2).unsqueeze(0).unsqueeze(0).float().to(node_type.device) #[1,1,dim/2]
            js = torch.pow(1.1, js) #[1,1,dim/2]
            B = torch.sin(js * node_score) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'linact':
            B = self.activation(self.B_lin(node_score)) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]


        X = H
        # print(X.size())
        edge_index, edge_type = A #edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
        _X = X.view(-1, X.size(2)).contiguous() #[`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        # print(_X.size())
        _node_type = node_type.view(-1).contiguous() #[`total_n_nodes`, ]
        _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0), -1).contiguous() #[`total_n_nodes`, dim]

        _X = self.mp_helper(_X, edge_index, edge_type, _node_type, _node_feature_extra)

        X = _X.view(node_type.size(0), node_type.size(1), -1) #[batch_size, n_node, dim]

        output = self.activation(self.Vh(H) + self.Vx(X))
        output = self.dropout(output)

        return output

class GNN(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, sent_dim,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02):
        super().__init__()
        self.init_range = init_range

        self.concept_emb = CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim,
                                               use_contextualized=False, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)
        self.svec2nvec = nn.Linear(sent_dim, concept_dim)
        # print("sent_dim:", sent_dim)
        # print("concept_dim:", concept_dim)
        # print(self.svec2nvec)
        self.concept_dim = concept_dim

        self.activation = GELU()

        self.gnn = GNN_Message_Passing(args, k=k, n_ntype=n_ntype, n_etype=n_etype,
                                        input_size=concept_dim, hidden_size=concept_dim, output_size=concept_dim, dropout=p_gnn)

        self.pooler = MultiheadAttPoolLayer(n_attention_head, sent_dim, concept_dim)

        self.fc = MLP(concept_dim + sent_dim + concept_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)

        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        if init_range > 0:
            self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, sent_vecs, concept_ids, node_type_ids, node_scores, adj_lengths, adj, emb_data=None, cache_output=False):
        """
        sent_vecs: (batch_size, dim_sent)
        concept_ids: (batch_size, n_node)
        adj: edge_index, edge_type
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_scores: (batch_size, n_node, 1)

        returns: (batch_size, 1)
        """
        # print("sent_vecs:", sent_vecs.size())
        gnn_input0 = self.activation(self.svec2nvec(sent_vecs)).unsqueeze(1) #(batch_size, 1, dim_node)
        gnn_input1 = self.concept_emb(concept_ids[:, 1:]-1, emb_data) #(batch_size, n_node-1, dim_node)
        gnn_input1 = gnn_input1.to(node_type_ids.device)
        # print(node_type_ids)
        # print(node_type_ids.size())
        # print("concept_ids_size:",concept_ids.size())
        # print(concept_ids[:, 1:])
        # print(gnn_input0.size())
        # print(gnn_input1.size())
        gnn_input = self.dropout_e(torch.cat([gnn_input0, gnn_input1], dim=1)) #(batch_size, n_node, dim_node)
        # print("gnn_input.size():", gnn_input.size())

        #Normalize node sore (use norm from Z)
        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(1)).float() #0 means masked out #[batch_size, n_node]
        node_scores = -node_scores
        node_scores = node_scores - node_scores[:, 0:1, :] #[batch_size, n_node, 1]
        node_scores = node_scores.squeeze(2) #[batch_size, n_node]
        node_scores = node_scores * _mask
        mean_norm  = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  #[batch_size, ]
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05) #[batch_size, n_node]
        node_scores = node_scores.unsqueeze(2) #[batch_size, n_node, 1]


        gnn_output = self.gnn(gnn_input, adj, node_type_ids, node_scores) #[batch_size, n_node, dim_node]
        # print("gnn_output.size():", gnn_output.size())

        # Z_vecs = gnn_output[:,0]   #(batch_size, dim_node)
        #
        # mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1) #1 means masked out
        #
        # mask = mask | (node_type_ids == 3) #pool over all KG nodes
        # mask[mask.all(1), 0] = 0  # a temporary solution to avoid zero node
        #
        # sent_vecs_for_pooler = sent_vecs
        # graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask)
        #
        # graph_vecs = 0.1 * graph_vecs
        #
        # if cache_output:
        #     self.concept_ids = concept_ids
        #     self.adj = adj
        #     self.pool_attn = pool_attn
        #
        # concat = self.dropout_fc(torch.cat((graph_vecs, sent_vecs, Z_vecs), 1))
        # logits = self.fc(concat)
        # return logits, pool_attn
        return gnn_output


class Binary_VQA_Model(nn.Module):
    def __init__(self, config):
        super(Binary_VQA_Model, self).__init__()

        embed_dim = config.embed_dim

        # self.tokenizer = AutoTokenizer.from_pretrained("../../../PMC-CLIP/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        # self.text_encoder = AutoModel.from_pretrained("../../../PMC-CLIP/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        self.tokenizer = AutoTokenizer.from_pretrained("./BioLinkBERT-large")
        self.text_encoder = AutoModel.from_pretrained("./BioLinkBERT-large", output_hidden_states=True)
        text_checkpoint = torch.load(config.pmcclip_pretrained)
        text_state_dict = text_checkpoint['state_dict']
        text_state_dict.pop('module.visual.attnpool.positional_embedding')
        self.text_encoder.load_state_dict({k.replace('module.', ''): v for k, v in text_state_dict.items()}, strict=False)

        # self.text_embed = nn.Sequential(nn.Linear(4096, embed_dim))
        # self.text_embed = nn.Sequential(nn.Linear(32000, embed_dim))
        # self.text_embed = nn.Sequential(nn.Linear(768, embed_dim))
        self.text_embed = nn.Sequential(nn.Linear(1024, embed_dim))  # biolinkbert
        self.gnn_embed = nn.Sequential(nn.Linear(100, embed_dim))  # biolinkbert

        # self.cls_id = 2  # [CLS]'s token id is 2, while it varies from tokenizers
        self.context_length = 256

        if config.image_encoder == "CLIP":
            self.image_encoder_name = "CLIP"
            configuration = CLIPVisionConfig(image_size=512)
            if config.clip_pretrained == "openai/clip-vit-base-patch32":
                self.image_encoder = CLIPVisionModel(configuration).from_pretrained(config.clip_pretrained)
                self.image_encoder.vision_model.embeddings = transformers.models.clip.modeling_clip.CLIPVisionEmbeddings(
                    configuration)
            else:
                self.image_encoder = CLIPVisionModel(configuration)
        elif config.image_encoder == "PMC_CLIP":
            self.image_encoder_name = "PMC_CLIP"
            self.image_encoder = PMC_CLIP(embed_dim=768)
            checkpoint = torch.load(config.pmcclip_pretrained)
            state_dict = checkpoint['state_dict']
            state_dict.pop('module.visual.attnpool.positional_embedding')
            self.image_encoder.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
            self.image_embed = nn.Sequential(nn.Linear(2048, embed_dim))

        self.qformer_query = nn.Parameter(torch.empty(32, 768))
        self.qformer_decoder_layer = nn.TransformerDecoderLayer(embed_dim, nhead=4, dim_feedforward=768, dropout=0.1,
                                                                activation='relu', norm_first=True)
        self.qformer_decoder_norm = nn.LayerNorm(embed_dim)
        self.qformer_decoder = nn.TransformerDecoder(self.qformer_decoder_layer, 12, self.qformer_decoder_norm)

        path = '../../../qagnn/data/ddb/sem_ent_emb_biolink.npy'
        cp_emb = [np.load(path)]
        cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
        concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
        k = config.k
        n_ntype = config.n_ntype
        n_etype = config.num_relation
        self.sent_dim = self.text_encoder.config.hidden_size
        n_concept = concept_num
        concept_in_dim = concept_dim
        n_attention_head = config.att_head_num
        fc_dim = config.fc_dim
        n_fc_layer = config.fc_layer_num
        p_emb = config.dropouti
        p_gnn = config.dropoutg
        p_fc = config.dropoutf
        init_range = config.init_range
        self.decoder = GNN(config, k, n_ntype, n_etype, self.sent_dim,
                             n_concept, config.gnn_dim, concept_in_dim, n_attention_head,
                             fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                             pretrained_concept_emb=cp_emb, freeze_ent_emb=True,
                             init_range=init_range)

        text_cfg = CLIPTextCfg
        self.transformer_width = text_cfg.width
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, text_cfg.width))

        self.text_projection = nn.Parameter(torch.empty(text_cfg.width, embed_dim))

        self.mlm_projection = nn.Parameter(torch.empty(text_cfg.width, text_cfg.vocab_size))
        self.softmax = nn.LogSoftmax(dim=-1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.img_special_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fusion_module = Transformer(
            width=text_cfg.width,
            layers=text_cfg.fusion_layers,
            heads=text_cfg.heads,
        )
        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)
        self.init_parameters()

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def init_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer_width ** -0.5)
        if self.mlm_projection is not None:
            nn.init.normal_(self.mlm_projection, std=self.transformer_width ** -0.5)

    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        #edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        #edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
        edge_type = edge_type_init
        # print(edge_type_init)
        # edge_type = torch.cat(edge_type_init, dim=0) #[total_E, ]
        return edge_index, edge_type[0]

    def forward(self, image, encoded_input_ids, encoded_attention_mask, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, cache_output=False):
        batchsize, _ = encoded_input_ids.shape
        if self.image_encoder_name == "CLIP":
            image_features = self.image_encoder(image).last_hidden_state[:, 1:, :]  # as key and value
        elif self.image_encoder_name == "PMC_CLIP":
            image_features = self.image_encoder(image) # [1, 2048, 16, 16]
            # print("image_features_size:", image_features.size())
            image_features = rearrange(image_features, 'b n h w -> (b h w) n') # [256(16*16), 2048]
            # print("image_features_size:", image_features.size())
            image_features = self.image_embed(image_features) # [256, 768]
            # print("image_features_size:", image_features.size())
            image_features = rearrange(image_features, '(b n) d -> b n d', b=batchsize) # [1, 256, 768]
            # print("image_features_size:", image_features.size())

        # print("qformer_query:", self.qformer_query)
        # print("qformer_query_size:", self.qformer_query.size())
        image_query_features = self.qformer_query.unsqueeze(0).expand(batchsize, -1, -1) #[1, 32, 768]
        # print("qformer_query:", self.qformer_query)
        # print("qformer_query_size:", self.qformer_query.size())
        # print("image_query_features:", image_query_features)
        # print("image_query_features_size:", image_query_features.size())
        # print("image_features:", image_features)
        # print("image_features_size:", image_features.size())
        image_features = self.qformer_decoder(image_query_features.transpose(0, 1),image_features.transpose(0, 1)).transpose(0, 1) # [batchsize, 32, 768]
        # print("image_features_size:", image_features.size())


        edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        adj = (edge_index, edge_type)  # edge_index: [2, total_E]   edge_type: [total_E, ]

        # question_features = self.text_encoder(input_ids=encoded_input_ids, attention_mask=encoded_attention_mask)[0]  # [1, 256, 1024]

        outputs = self.text_encoder(input_ids=encoded_input_ids, attention_mask=encoded_attention_mask)
        all_hidden_states = outputs[-1]
        hidden_states = all_hidden_states[-1] #[1, 256, 1024]

        sent_vecs = self.text_encoder.pooler(hidden_states) #[1, 1024]
        gnn_features = self.decoder(sent_vecs,
                                    concept_ids,
                                    node_type_ids, node_scores, adj_lengths, adj,
                                    emb_data=None, cache_output=cache_output) # [1, 100, 100]
        gnn_features = rearrange(gnn_features, 'b n d -> (b n) d').float() #[100, 100]
        gnn_features = self.gnn_embed(gnn_features.to('cuda')) #[100, 768]
        graph_features = rearrange(gnn_features, '(b n)d -> b n d', b=batchsize) #[1, 100, 768]

        question_features = outputs[0]
        question_features = rearrange(question_features, 'b n d -> (b n) d').float()
        # question_features = rearrange(question_features,'b n d -> (b n) d').float()
        question_features = self.text_embed(question_features.to('cuda'))
        x = rearrange(question_features, '(b n)d -> b n d', b=batchsize) #[1, 256, 768]

        B, _len, _dim = x.shape
        img_special_tokens = self.img_special_token.expand(B, -1, -1)  # [1, 1, 768]

        x = torch.cat([x, graph_features, img_special_tokens, image_features], dim=1)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.fusion_module(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, :-133, :]  # Remove token [graph, img_special_token, img]

        out = self.softmax(x @ self.mlm_projection / 2)  # [batch_size=128, n_ctx=77, vocab_size=49409]
        return out

