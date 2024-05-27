import torch
from tqdm import tqdm
import numpy as np
import transformers
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AutoModelForCausalLM

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
if __name__ == '__main__':
    repo_root = '.'

    # model = AutoModelForCausalLM.from_pretrained(
    #     "./LLAMA_PMC7b",
    #     load_in_8bit=True,
    #     device_map="auto",
    #     trust_remote_code=True,
    # )
    # model.eval()
    #
    # tokenizer = AutoTokenizer.from_pretrained("./LLAMA_PMC7b")
    # tokenizer.pad_token_id = 0
    # tokenizer.eos_token_id = 1
    # # model = AutoModel.from_pretrained("./LLAMA_PMC7b")
    #
    # with open("../qagnn/data/ddb/sem_vocab.txt") as f:
    #     names = [line.strip() for line in f]
    # embs = []
    # tensors = tokenizer(names, padding=True, truncation=True, return_tensors="pt")
    # with torch.no_grad():
    #     for i, j in enumerate(tqdm(names)):
    #         outputs = model(input_ids=tensors["input_ids"][i:i + 1],
    #                              attention_mask=tensors['attention_mask'][i:i + 1])
    #         print(outputs[1])
    #         out = np.array(outputs[1].squeeze().tolist()).reshape((1, -1))
    #         embs.append(out)
    # embs = np.concatenate(embs)
    # np.save(f"{repo_root}/Slake1.0_new/KG/kg_LLM_sem_emb.npy", embs)

    tokenizer = AutoTokenizer.from_pretrained("../PMC-CLIP/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = AutoModel.from_pretrained("../PMC-CLIP/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    with open("../qagnn/data/ddb/sem_vocab.txt") as f:
        names = [line.strip() for line in f]
    embs = []
    tensors = tokenizer(names, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        for i, j in enumerate(tqdm(names)):
            outputs = model(input_ids=tensors["input_ids"][i:i + 1],attention_mask=tensors['attention_mask'][i:i + 1])
            out = np.array(outputs[1].squeeze().tolist()).reshape((1, -1))
            embs.append(out)
    embs = np.concatenate(embs)
    np.save(f"{repo_root}/Slake1.0_new/KG/pubmedbert_sem_emb.npy", embs)