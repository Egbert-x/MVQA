import json
from tqdm import tqdm
import csv
import json
import configparser
import time
import networkx as nx

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
    kg_root = f'{repo_root}/Slake1.0_new/KG'
    with open("./Slake1.0_new/KG/kg_vocab.txt", "r", encoding="gbk") as fin:
        idx2cui = [c.strip() for c in fin]
    cui2idx = {c: i for i, c in enumerate(idx2cui)}
    with open("./Slake1.0_new/KG/kg_rels.txt", "r", encoding="gbk") as fin:
        id2relation = [c.strip() for c in fin]
    relation2id = {r: i for i, r in enumerate(id2relation)}
    print(id2relation)
    # 创建有向多图
    print("generating graph of SLAKE using newly extracted vocab list...")
    graph = nx.MultiDiGraph()
    attrs = set()
    with open(f"{kg_root}/en_disease.csv", "r", encoding="ascii", errors="ignore") as f:
        reader = csv.reader(f, skipinitialspace=True)
        for i, row in enumerate(reader):
            if i > 0:
                weight = 1.
                row_sep = row[0].split('#')
                # print(row_sep[1])
                rel = relation2id[row_sep[1]]  # 转成id形式
                if row_sep[0] in idx2cui:
                    subj = cui2idx[row_sep[0]]
                    obj_list = []
                    obj_list.append(cui2idx[row_sep[2]])
                    for j in range(0, len(row)):
                        if j > 0:
                            if row[j] in idx2cui:
                                obj_list.append(cui2idx[row[j]])
                    for obj in obj_list:
                        if (subj, obj, rel) not in attrs:
                            # print("type3 add")
                            graph.add_edge(subj, obj, rel=rel, weight=weight)
                            attrs.add((subj, obj, rel))
                            graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)
                            attrs.add((obj, subj, rel + len(relation2id)))

    with open(f"{kg_root}/en_organ.csv", "r", encoding="ascii", errors="ignore") as f:
        reader = csv.reader(f, skipinitialspace=True)
        for i, row in enumerate(reader):
            if i > 0:
                weight = 1.
                row_sep = row[0].split('#')
                # print(row_sep[1])
                rel = relation2id[row_sep[1]]  # 转成id形式
                if row_sep[0] in idx2cui:
                    subj = cui2idx[row_sep[0]]
                    obj_list = []
                    obj_list.append(cui2idx[row_sep[2]])
                    for j in range(0, len(row)):
                        if j > 0:
                            if row[j] in idx2cui:
                                obj_list.append(cui2idx[row[j]])
                    for obj in obj_list:
                        if (subj, obj, rel) not in attrs:
                            # print("type3 add")
                            graph.add_edge(subj, obj, rel=rel, weight=weight)
                            attrs.add((subj, obj, rel))
                            graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)
                            attrs.add((obj, subj, rel + len(relation2id)))


    output_path = f"{kg_root}/kg.graph"
    nx.write_gpickle(graph, output_path)
    print(len(attrs))