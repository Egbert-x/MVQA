import csv
import os
import time
import json
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# 分离多个cui
def separate_semmed_cui(semmed_cui: str) -> list:
    """
    separate semmed cui with | by perserving the replace the numbers after |
    `param`:
        semmed_cui: single or multiple semmed_cui separated by |
    `return`:
        sep_cui_list: list of all separated semmed_cui
    """
    sep_cui_list = []
    sep = semmed_cui.split("|")
    first_cui = sep[0]
    sep_cui_list.append(first_cui)
    ncui = len(sep)
    for i in range(ncui - 1):
        last_digs = sep[i + 1]
        len_digs = len(last_digs)
        if len_digs < 8:
            sep_cui = first_cui[:8 - len(last_digs)] + last_digs
            sep_cui_list.append(sep_cui)
    return sep_cui_list

if __name__ == '__main__':
    repo_root = '.'
    semmed_root = '../qagnn/data/ddb'
    cui_name_lst = {}
    # sem_relas = {}
    # id = 0
    with open("../data/semmedVER43_2022_R_PREDICATION.csv", "r", encoding="gb18030", errors="ignore") as f:
        reader = csv.reader(f, skipinitialspace=True)
        for i, row in enumerate(reader):
            if row[4].startswith("C"):
                cui_subj = row[4]
                name_subj = row[5]
                # print("single_subj")
                # print(cui_subj)
                if len(cui_subj) != 8:
                    cui_subj_list = separate_semmed_cui(cui_subj)
                    # print("subj")
                    # print(cui_subj_list)
                    # time.sleep(5)
                    for cui in cui_subj_list:
                        cui_name_lst[cui] = name_subj
                else:
                    cui_name_lst[cui_subj] = name_subj
            if row[8].startswith("C"):
                cui_obj = row[8]
                name_obj = row[9]
                # print("single_obj")
                # print(cui_obj)
                if len(cui_obj) != 8:
                    cui_obj_list = separate_semmed_cui(cui_obj)
                    # print("obj")
                    # print(cui_obj_list)
                    # time.sleep(5)
                    for cui in cui_obj_list:
                        cui_name_lst[cui] = name_obj
                else:
                    cui_name_lst[cui_obj] = name_obj


    print("read done!")

    json_str = json.dumps(cui_name_lst)
    with open("../qagnn/data/ddb/sem_cui_name_lst.json", 'w') as json_file:
        json_file.write(json_str)
