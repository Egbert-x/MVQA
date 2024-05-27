import argparse
import os
import csv
import json
import math
import numpy as np
import tqdm.auto as tqdm
from typing import Optional
import difflib
import transformers
from transformers import Trainer
from dataclasses import dataclass, field
from torch import nn
from torch.utils.data import DataLoader
import torch
# from tensorboardX import SummaryWriter
from torch.nn import functional as F

from dataset.slake_dataset_gnn import SLAKE_VQA_Dataset
# from models.llama.vqa_model_gnn import Binary_VQA_Model
from models.llama.vqa_model_gnn_v2 import Binary_VQA_Model

@dataclass
class ModelArguments:
    embed_dim: Optional[int] = field(default=768)
    pretrained_tokenizer: Optional[str] = field(default="../../LLAMA_PMC7b")
    pretrained_model: Optional[str] = field(default="../../LLAMA_PMC7b")
    image_encoder: Optional[str] = field(default="CLIP")
    pmcclip_pretrained: Optional[str] = field(default="./models/pmc-clip/checkpoint.pt")
    clip_pretrained: Optional[str] = field(default="openai/clip-vit-base-patch32")
    ckp: Optional[str] = field(default="./Results/VQA_lora_noclip/vqa/checkpoint-6500")
    num_relation: Optional[int] = field(default=26)
    k: Optional[int] = field(default=4)  # num of gnn layers
    n_ntype: Optional[int] = field(default=4)  # node type
    gnn_dim: Optional[int] = field(default=100)
    att_head_num: Optional[int] = field(default=2)  # number of attention heads
    fc_dim: Optional[int] = field(default=100)  # number of FC hidden units
    fc_layer_num: Optional[int] = field(default=0)  # number of FC layers
    dropouti: Optional[float] = field(default=0.2)  # dropout for embedding layer
    dropoutg: Optional[float] = field(default=0.2)  # dropout for GNN layers
    dropoutf: Optional[float] = field(default=0.2)  # dropout for fully-connected layers
    init_range: Optional[float] = field(default=0.02)  # stddev when initializing with normal distribution


@dataclass
class DataArguments:
    is_blank: Optional[bool] = field(default=False)
    image_res: Optional[int] = field(default=512)
    img_root_dir: str = field(default='../../Slake1.0_new/imgs/', metadata={"help": "Path to the training data."})
    Train_csv_path: str = field(default='../../Slake1.0_new/train.csv', metadata={"help": "Path to the training data."})
    Test_csv_path: str = field(default='../../Slake1.0_new/test.csv', metadata={"help": "Path to the training data."})
    Train_adj_path: str = field(default='../../Slake1.0_new/graph/train_sem.graph.adj.pk', metadata={"help": "Path to the training data."})
    Test_adj_path: str = field(default='../../Slake1.0_new/graph/test_sem.graph.adj.pk',metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="./Results")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    logging_dir: Optional[str] = field(default="./logs")
    logging_steps: Optional[int] = field(default=50)


def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()


def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0

    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)

        # If the current string is more similar than the previous most similar string, update the variables
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity

    # Return the index of the most similar string
    return most_similar_index

def find_similar(source_str, target_str):
    is_similar = 0
    similarity = str_similarity(source_str, target_str)
    if similarity > 0.5:
        is_similar = 1
    return is_similar

def get_generated_texts(label, outputs, tokenizer):
    # 1,256
    # print("label:",label)
    # print("outputs_before:", outputs)
    outputs = outputs[label != 0][1:-1]
    # print("outputs_after:", outputs)
    generated_text = tokenizer.decode(outputs)
    return generated_text


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("Setup Data")
    Test_dataset = SLAKE_VQA_Dataset(data_args.Test_csv_path, data_args.Test_adj_path, data_args.img_root_dir, data_args.image_res,is_train=False)
    # batch size should be 1
    Test_dataloader = DataLoader(
        Test_dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=None,
        drop_last=False,
    )

    print("Setup Model")
    ckp = model_args.ckp + '/pytorch_model.bin'
    model = Binary_VQA_Model(model_args)
    model.load_state_dict(torch.load(ckp, map_location='cpu'), strict=False)

    ACC = 0
    cc = 0

    print("Start Testing")

    model = model.to('cuda')
    model.eval()
    with open(os.path.join(training_args.output_dir, 'result.csv'), mode='w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Figure_path','Question','Pred','Answer','Correct'])
        for sample in tqdm.tqdm(Test_dataloader):
            img_path = sample['image_path']
            image = sample['image'].to('cuda')
            label = sample['label'].to('cuda')[:, 0, :]
            question_inputids = sample['encoded_input_ids'].to('cuda')[:, 0, :]
            question_attenmask = sample['encoded_attention_mask'].to('cuda')[:, 0, :]
            concept_ids = sample['concept_ids'].to('cuda')
            node_type_ids = sample['node_type_ids'].to('cuda')
            node_scores = sample['node_scores'].to('cuda')
            adj_lengths = sample['adj_lengths'].to('cuda')
            edge_index = sample['edge_index'].to('cuda')
            edge_type = sample['edge_type'].to('cuda')
            with torch.no_grad():
                outputs = model(image, question_inputids, question_attenmask, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type)
                # print("outputs_before:", outputs)

            generated_texts = get_generated_texts(label, outputs.argmax(-1), Test_dataset.tokenizer)
            Question = sample['question'][0]
            Answer = sample['Answer'][0]
            # print("label:", label)
            # print("Question:", Question)
            # print("Answer:",Answer)
            # print("generated_texts:", generated_texts)
            is_similar = find_similar(Answer.lower(), generated_texts)
            corret = 0
            if is_similar == 1:
                ACC = ACC + 1
                corret = 1
            writer.writerow([img_path, Question, generated_texts, Answer, corret])
            cc = cc + 1
        print(ACC / cc)
        writer.writerow([ACC / cc])


if __name__ == "__main__":
    main()

