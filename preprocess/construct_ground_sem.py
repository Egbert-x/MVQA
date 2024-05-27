import csv
import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# 构造ground数据集
repo_root = '.'
# kg_root = f'{repo_root}/Slake1.0_new'
kg_root = f'{repo_root}/PMC-VQA'
# kg_root = f'{repo_root}/VQA-RAD/data'

umls_to_ddb = {}
umls_to_ddb_reverse = {}
with open("../data/semmedVER43_2022_R_PREDICATION.csv", "r", encoding="gb18030", errors="ignore") as f:
    for row in csv.reader(f, skipinitialspace=True):
        umls_to_ddb[row[4]] = row[5]
        umls_to_ddb[row[8]] = row[9]
        umls_to_ddb_reverse[row[5]] = row[4]
        umls_to_ddb_reverse[row[9]] = row[8]
def map_to_ddb(ent_obj):
    res = []
    ignore = []
    for ent_cand in ent_obj['linking_results']:
        CUI  = ent_cand['Concept ID']
        name = ent_cand['Canonical Name']
        if CUI in umls_to_ddb and name == umls_to_ddb[CUI]:
            ddb_cid = CUI
            res.append((ddb_cid, name))
            # if CUI == "C1514243" and name == "Positive Predictive Value of Diagnostic Test":
            #     print("Positive Predictive Value of Diagnostic Test find!")
        else:
            ignore.append((CUI, name))
    return res

def map_to_ddb_ac(ent_obj, ans):
    res = []
    for ent_cand in ent_obj['linking_results']:
        CUI  = ent_cand['Concept ID']
        name = ent_cand['Canonical Name']
        if ans in umls_to_ddb_reverse and umls_to_ddb_reverse[ans].startswith("C"):
            ddb_cid = umls_to_ddb_reverse[ans]
            res.append((ddb_cid, ans))
            break
        if CUI in umls_to_ddb and name == umls_to_ddb[CUI]:
            ddb_cid = CUI
            res.append((ddb_cid, name))
            # if CUI == "C1514243" and name == "Positive Predictive Value of Diagnostic Test":
            #     print("Positive Predictive Value of Diagnostic Test find!")
            break
    return res

def process(fname):
    with open(f"{kg_root}/{fname}_linked_md_051.json") as fin:
        stmts = [json.loads(line) for line in fin]
    with open(f"{kg_root}/grounded/{fname}_sem_grounded_md_051.json", 'w') as fout:
        i = 1
        for stmt in tqdm(stmts):
            sent = stmt['Question']
            qc = []
            qc_names = []
            for ent_obj in stmt['question_ents']:
                res = map_to_ddb(ent_obj)
                for elm in res:
                    ddb_cid, name = elm
                    qc.append(ddb_cid)
                    qc_names.append(name)
            ans = stmt['Answer']
            ac = []
            ac_names = []
            for ent_obj in stmt['answer_ents']:
                # res = map_to_ddb_ignore(i, cid, ent_obj)
                res = map_to_ddb_ac(ent_obj, ans)
                for elm in res:
                    ddb_cid, name = elm
                    ac.append(ddb_cid)
                    ac_names.append(name)
            out = {'sent': sent, 'ans': ans, 'qc': qc, 'qc_names': qc_names, 'ac': ac, 'ac_names': ac_names}
            print(json.dumps(out), file=fout)

            i+=1

os.system(f'mkdir -p {kg_root}/grounded')
for fname in ["test", "train"]:
    process(fname)