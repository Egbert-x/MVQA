import argparse
import os
import json
import math
import tqdm.auto as tqdm
from typing import Optional
import transformers
from Dataset.VQA_RAD_Dataset_triples import VQA_RAD_Dataset_triples
from models.QA_model import QA_model
from transformers import Trainer
from dataclasses import dataclass, field
import os
from torch.utils.data import DataLoader
import torch
import numpy as np
import difflib
import csv


@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="../../../PMC-VQA/LLAMA_PMC7b")
    ckp: Optional[str] = field(default="./Results/QA_no_pretrain_no_aug/VQA_RAD/checkpoint-16128")
    checkpointing: Optional[bool] = field(default=False)
    ## Q_former ##
    N: Optional[int] = field(default=12)
    H: Optional[int] = field(default=8)
    img_token_num: Optional[int] = field(default=32)

    ## Basic Setting ##
    voc_size: Optional[int] = field(default=32000)
    hidden_dim: Optional[int] = field(default=4096)

    ## Image Encoder ##
    Vision_module: Optional[str] = field(default='PMC-CLIP')
    visual_model_path: Optional[str] = field(default='../../../PMC-VQA/src/MedVInT_TE/models/pmc_clip/checkpoint.pt')

    ## Peft ##
    is_lora: Optional[bool] = field(default=True)
    peft_mode: Optional[str] = field(default="lora")
    lora_rank: Optional[int] = field(default=8)


@dataclass
class DataArguments:
    img_dir: str = field(default='../../../PMC-VQA/VQA-RAD/VQA_RAD Image Folder/', metadata={"help": "Path to the training data."})
    # Test_csv_path: str = field(default='../../../PMC-VQA/VQA-RAD/data/test/test_close.csv', metadata={"help": "Path to the training data."})
    Test_csv_path: str = field(default='../../../PMC-VQA/VQA-RAD/data/test/test.csv', metadata={"help": "Path to the training data."})
    Test_triples_path: str = field(default='../../../PMC-VQA/VQA-RAD/data/triples/test_sem.triples.json', metadata={"help": "Path to the training data."})
    tokenizer_path: str = field(default='../../../PMC-VQA/LLAMA_PMC7b', metadata={"help": "Path to the training data."})
    trier: int = field(default=0)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="./Results")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()


def find_similar(source_str, target_str):
    is_similar = 0
    similarity = str_similarity(source_str, target_str)
    if similarity > 0.5:
        is_similar = 1
    return is_similar


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


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("Setup Data")
    row_count = 0
    # if os.path.exists('result_final'+str(data_args.trier)+'.csv'):

    #     with open('result_final'+str(data_args.trier)+'.csv', 'r') as file:
    #         reader = csv.reader(file)
    #         row_count = sum(1 for row in reader)-1

    Test_dataset = VQA_RAD_Dataset_triples(data_args.img_dir, data_args.Test_csv_path,
                                                 data_args.Test_triples_path, data_args.tokenizer_path, mode='Test',
                                                 text_type='blank', start=row_count)

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
    
    # Test_dataset_close = VQA_RAD_Dataset_triples(data_args.img_dir, data_args.Test_csv_path, data_args.Test_triples_path, data_args.tokenizer_path, mode='Test', text_type='blank', start=row_count)
    # 
    # # batch size should be 1
    # Test_dataloader_close = DataLoader(
    #     Test_dataset_close,
    #     batch_size=1,
    #     num_workers=1,
    #     pin_memory=True,
    #     sampler=None,
    #     shuffle=False,
    #     collate_fn=None,
    #     drop_last=False,
    # )
    # Test_dataset_open = VQA_RAD_Dataset_triples(data_args.img_dir, data_args.Test_csv_path.replace('close.csv', 'open.csv'), data_args.Test_triples_path, data_args.tokenizer_path, mode='Test', text_type='blank', start=row_count)
    # 
    # # batch size should be 1
    # Test_dataloader_open = DataLoader(
    #     Test_dataset_open,
    #     batch_size=1,
    #     num_workers=1,
    #     pin_memory=True,
    #     sampler=None,
    #     shuffle=False,
    #     collate_fn=None,
    #     drop_last=False,
    # )

    print("Setup Model")
    ckp = model_args.ckp + '/pytorch_model.bin'
    print(ckp)
    model = QA_model(model_args)
    model.load_state_dict(torch.load(ckp, map_location='cpu'), strict=False)

    ACC = 0
    cc = 0
    model = model.to('cuda')
    model.eval()
    # Test_dataset.tokenizer.padding_side = "left"

    with open(os.path.join(training_args.output_dir, 'result.csv'), mode='w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Figure_path', 'Question', 'Pred', 'Label', 'Correct'])
        for sample in tqdm.tqdm(Test_dataloader):
            input_ids = Test_dataset.tokenizer(sample['input_ids'], return_tensors="pt").to('cuda')
            input_ids['input_ids'][0][0] = 1
            images = sample['images'].to('cuda')
            with torch.no_grad():
                generation_ids = model.generate_long_sentence(input_ids['input_ids'], images)
            generated_texts = Test_dataset.tokenizer.batch_decode(generation_ids, skip_special_tokens=True)
            label = sample['labels']
            img_path = sample['img_path']
            pred = generated_texts
            # print("label:", label)
            # print("pred:", pred)
            corret = 0
            is_similar = find_similar(generated_texts, label)
            if is_similar == 1:
                ACC = ACC + 1
                corret = 1
            writer.writerow([img_path, sample['input_ids'], pred, label, corret])
            cc = cc + 1
        writer.writerow(['ACC', 'cc', 'ACC / cc'])
        writer.writerow([ACC, cc, ACC / cc])

    print("acc:", ACC / cc)

    # ACC = 0
    # cc = 0
    # model = model.to('cuda')
    # model.eval()
    # # Test_dataset.tokenizer.padding_side = "left"
    # 
    # with open(os.path.join(training_args.output_dir, 'result_close.csv'), mode='w') as outfile:
    #     writer = csv.writer(outfile)
    #     writer.writerow(['Figure_path', 'Question', 'Pred', 'Label', 'Correct'])
    #     for sample in tqdm.tqdm(Test_dataloader_close):
    #         input_ids = Test_dataset_close.tokenizer(sample['input_ids'], return_tensors="pt").to('cuda')
    #         input_ids['input_ids'][0][0] = 1
    #         images = sample['images'].to('cuda')
    #         with torch.no_grad():
    #             generation_ids = model.generate_long_sentence(input_ids['input_ids'], images)
    #         generated_texts = Test_dataset_close.tokenizer.batch_decode(generation_ids, skip_special_tokens=True)
    #         label = sample['labels']
    #         img_path = sample['img_path']
    #         pred = generated_texts
    #         # print("label:", label)
    #         # print("pred:", pred)
    #         corret = 0
    #         is_similar = find_similar(generated_texts, label)
    #         if is_similar == 1:
    #             ACC = ACC + 1
    #             corret = 1
    #         writer.writerow([img_path, sample['input_ids'], pred, label, corret])
    #         cc = cc + 1
    #     writer.writerow(['ACC', 'cc', 'ACC / cc'])
    #     writer.writerow([ACC, cc, ACC / cc])
    # 
    # print("close acc:", ACC / cc)
    # 
    # ACC = 0
    # cc = 0
    # 
    # with open(os.path.join(training_args.output_dir, 'result_open.csv'), mode='w') as outfile:
    #     writer = csv.writer(outfile)
    #     writer.writerow(['Figure_path', 'Question', 'Pred', 'Label', 'Correct'])
    #     for sample in tqdm.tqdm(Test_dataloader_open):
    #         input_ids = Test_dataset_open.tokenizer(sample['input_ids'], return_tensors="pt").to('cuda')
    #         input_ids['input_ids'][0][0] = 1
    #         images = sample['images'].to('cuda')
    #         with torch.no_grad():
    #             generation_ids = model.generate_long_sentence(input_ids['input_ids'], images)
    #         generated_texts = Test_dataset_open.tokenizer.batch_decode(generation_ids, skip_special_tokens=True)
    #         label = sample['labels']
    #         img_path = sample['img_path']
    #         pred = generated_texts
    #         # print("pred:", pred)
    #         corret = 0
    #         is_similar = find_similar(generated_texts, label)
    #         if is_similar == 1:
    #             ACC = ACC + 1
    #             corret = 1
    #         writer.writerow([img_path, sample['input_ids'], pred, label, corret])
    #         cc = cc + 1
    #     writer.writerow(['ACC', 'cc', 'ACC / cc'])
    #     writer.writerow([ACC, cc, ACC / cc])
    # print("open acc:", ACC / cc)

if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

# CUDA_VISIBLE_DEVICES=2  python test.py