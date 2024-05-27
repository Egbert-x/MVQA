import json
from tqdm import tqdm
import csv
import json
import pickle
import configparser
import time
import os
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from multiprocessing import Pool


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


def concepts2adj(node_ids):
    global id2relation
    # print(len(node_ids))
    # print(node_ids)
    cids = np.array(node_ids, dtype=np.int32)
    n_rel = len(id2relation)
    n_node = cids.shape[0]
    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if cpnet.has_edge(s_c, t_c):
                for e_attr in cpnet[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        adj[e_attr['rel']][s][t] = 1
    adj = coo_matrix(adj.reshape(-1, n_node))
    return adj, cids

def concepts2triple(node_ids):
    global id2relation
    # print(len(node_ids))
    # print(node_ids)
    cids = np.array(node_ids, dtype=np.int32)
    n_rel = len(id2relation)
    n_node = cids.shape[0]
    triples = []
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if cpnet.has_edge(s_c, t_c):
                for e_attr in cpnet[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        head_cui = idx2cui[s_c]
                        rel = id2relation[e_attr['rel']]
                        tail_cui = idx2cui[t_c]
                        head = cui_name_pair[head_cui]
                        tail = cui_name_pair[tail_cui]
                        triple = []
                        triple.append(head)
                        triple.append(rel)
                        triple.append(tail)
                        triples.append(triple)
    return triples

#
def concepts_to_adj_matrices_2hop_all_pair(data):
    qc_ids, ac_ids = data
    # set():閺冪姴绨稉宥夊櫢婢跺秴鍘撶槐鐘绘肠
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    # print(cpnet_simple.nodes)
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    # print(len(extra_nodes))
    # print(extra_nodes)
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return {'adj': adj, 'concepts': concepts, 'qmask': qmask, 'amask': amask, 'cid2score': None}

def concepts_to_triples(data):
    qc_ids, ac_ids = data
    # set():閺冪姴绨稉宥夊櫢婢跺秴鍘撶槐鐘绘肠
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    # print(cpnet_simple.nodes)
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    # print(len(extra_nodes))
    # print(extra_nodes)
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    triples = concepts2triple(schema_graph)
    # print("triples:", triples)
    return {'triples': triples}


def generate_adj_data_from_grounded_concepts(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes):
    # This function will save
    # (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
    # (2) concepts ids
    # (3) qmask that specifices whether a node is a question concept
    # (4) amask that specifices whether a node is a answer concept
    # to the output path in python pickle format
    global cui2idx, idx2cui, relation2id, id2relation, cpnet_simple, cpnet, cui_name_pair

    qa_data = []
    with open(grounded_path, 'r', encoding='utf-8') as fin:
        i = 1
        num_qc = 0
        num_ac = 0
        for line in fin:
            dic = json.loads(line)
            q_ids = set(cui2idx[c] for c in dic['qc'])
            # if not q_ids:
            #     q_ids = {cui2idx['C0241028']}
            #     num_qc += 1
            a_ids = set(cui2idx[c] for c in dic['ac'])
            # if not a_ids:
            #     a_ids = {cui2idx['C0007561']}
            #     num_ac += 1
            # 瀹割噣娉?
            q_ids = q_ids - a_ids
            qa_data.append((q_ids, a_ids))
            i += 1

    # with Pool(num_processes) as p:
    #     res = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair, qa_data), total=len(qa_data)))

    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(concepts_to_triples, qa_data), total=len(qa_data)))

    print(len(res))

    # lens = [len(e['concepts']) for e in res]
    # print('mean #nodes', int(np.mean(lens)), 'med', int(np.median(lens)), '5th', int(np.percentile(lens, 5)), '95th',
    #       int(np.percentile(lens, 95)))
    #
    # with open(output_path, 'wb') as fout:
    #     pickle.dump(res, fout)

    json_str = json.dumps(res)
    with open(output_path, 'w') as json_file:
        json_file.write(json_str)
    print(f'triples data saved to {output_path}')

    return num_qc, num_ac


if __name__ == '__main__':
    repo_root = '.'
    # kg_root = f'{repo_root}/Slake1.0_new'
    kg_root = f'{repo_root}/PMC-VQA'
    # kg_root = f'{repo_root}/VQA-RAD/data'
    merged_relations = ["process_of", "affects", "augments", "causes", "diagnoses", "interacts_with", "part_of",
                        "precedes", "predisposes", "produces", "isa"]
    with open("../qagnn/data/ddb/sem_cuis.txt", "r", encoding="gbk") as fin:
        idx2cui = [c.strip() for c in fin]
    cui2idx = {c: i for i, c in enumerate(idx2cui)}
    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}
    print(id2relation)

    f = open('../qagnn/data/ddb/sem_cui_name_lst.json', 'r')
    content = f.read()
    cui_name_pair = json.loads(content)
    f.close()
    print(len(cui_name_pair))

    graph = nx.read_gpickle("../qagnn/data/ddb/sem.graph")

    print("graph done!")
    # KG:閻儴鐦戦崶鎹愭皑ddb.graph
    cpnet = graph
    # cpnet_simple娑撶儤妫ら崥鎴濇禈
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


    os.system(f'mkdir -p {kg_root}/triples')
    for fname in ["test", "train"]:
        grounded_path = f"{kg_root}/grounded/{fname}_sem_grounded_md_051.json"
        kg_path = "../qagnn/data/ddb/sem.graph"
        kg_vocab_path = "../qagnn/data/ddb/sem_cuis.txt"
        output_path = f"{kg_root}/triples/{fname}_sem.triples.json"

        numqc, numac = generate_adj_data_from_grounded_concepts(grounded_path, kg_path, kg_vocab_path, output_path, 10)
        print("num_qc: ", numqc, "num_ac: ",numac)