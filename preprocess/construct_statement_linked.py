import spacy
import scispacy
from scispacy.linking import EntityLinker
import json
from tqdm import tqdm
import csv

repo_root = '.'
# kg_root = f'{repo_root}/Slake1.0_new'
# kg_root = f'{repo_root}/PMC-VQA'
kg_root = f'{repo_root}/VQA-RAD/data'

def load_entity_linker(threshold=0.90):
    nlp = spacy.load("en_core_sci_md")
    # linker = EntityLinker(
    #     resolve_abbreviations=True,
    #     name="umls",
    #     threshold=threshold)
    # nlp.add_pipe(linker)
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    linker = nlp.get_pipe("scispacy_linker")
    return nlp, linker

# nlp, linker = load_entity_linker()

def entity_linking_to_umls(sentence, nlp, linker):
    doc = nlp(sentence)
    entities = doc.ents
    all_entities_results = []
    for mm in range(len(entities)):
        entity_text = entities[mm].text
        entity_start = entities[mm].start
        entity_end = entities[mm].end
        all_linked_entities = entities[mm]._.kb_ents
        all_entity_results = []
        for ii in range(len(all_linked_entities)):
            curr_concept_id = all_linked_entities[ii][0]
            curr_score = all_linked_entities[ii][1]
            curr_scispacy_entity = linker.kb.cui_to_entity[all_linked_entities[ii][0]]
            curr_canonical_name = curr_scispacy_entity.canonical_name
            curr_TUIs = curr_scispacy_entity.types
            curr_entity_result = {"Canonical Name": curr_canonical_name, "Concept ID": curr_concept_id,
                                  "TUIs": curr_TUIs, "Score": curr_score}
            all_entity_results.append(curr_entity_result)
        curr_entities_result = {"text": entity_text, "start": entity_start, "end": entity_end,
                                "start_char": entities[mm].start_char, "end_char": entities[mm].end_char,
                                "linking_results": all_entity_results}
        all_entities_results.append(curr_entities_result)
    return all_entities_results

# def process(input):
#     nlp, linker = load_entity_linker()
#     # stmts = list(input[0])
#     stmts = input
#     # print(len(stmts))
#     for stmt in tqdm(stmts):
#         # print(stmt)
#         if stmt['q_lang'] == "zh":
#             continue
#         stem = stmt['question']
#         stem = stem[:3500]
#         text = stmt['answer']
#         # print("question: ", stem)
#         # print("answer: ", stmt['answer'])
#         stmt['question_ents'] = entity_linking_to_umls(stem, nlp, linker)
#         stmt['answer_ents'] = entity_linking_to_umls(text, nlp, linker)
#     return stmts

def process(input):
    nlp, linker = load_entity_linker()
    # stmts = list(input[0])
    stmts = input
    # print(len(stmts))
    for stmt in tqdm(stmts):
        # print(stmt)
        stem = stmt['question']
        stem = stem[:3500]
        text = stmt['answer']
        # print("question: ", stem)
        # print("answer: ", stmt['answer'])
        stmt['question_ents'] = entity_linking_to_umls(stem, nlp, linker)
        stmt['answer_ents'] = entity_linking_to_umls(text, nlp, linker)
    return stmts

for fname in ["test", "train"]:
    with open(f"{kg_root}/{fname}.json") as fin:
        stmts = json.load(fin)
        # stmts = [json.loads(line) for line in fin]
        res = process(stmts)
    with open(f"{kg_root}/{fname}_linked_md_051.json", 'w') as fout:
        for dic in res:
            print (json.dumps(dic), file=fout)
