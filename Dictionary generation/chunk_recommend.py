# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 14:00:00 2020

@author: wwr
"""
import os
from nltk.parse.stanford import StanfordParser
from nltk.tree import Tree
import re
import xlrd
import json
from tqdm import tqdm
import numpy as np
import json
from copy import deepcopy
import copy

aft_rule = {"^(VBD\sVB[DN]\sIN)\sIN":4,"^VBD\sVB[DN]":2,"^VBD\sRB\sVB[DN]":3}
# be carried out,was made,

bef_rule = {"NN\sVB.*$":2,"VB[DGN]\sIN$":2,"VB[DGN]\sTO$":2,"IN\sNN.*\sIN$":3}#,"VBG$":1,"IN$":1
# Following,prepared by,by method of

chunk_limit = ["°C",'h','K']

def corpus_preprocess(sentences):
    new_sents = list()
    for sent in sentences:
        sent = re.sub("\s+", " ", sent.lower())
        sent = sent.replace("\n","")
        new_sents.append(sent.strip())
    return new_sents

def seed_read(seed_path):
    lower_seeds = list()
    xls_sp = xlrd.open_workbook(seed_path)
    sht_sp = xls_sp.sheet_by_index(0)
    cols_sp = sht_sp.col_values(0)
    seeds = list(set(cols_sp))
    for seed in seeds:
        lower_seeds.append(seed.lower())
    return lower_seeds

def sentences_read(sentence_path):
    lower_chunks = list()
    xls_sp = xlrd.open_workbook(sentence_path)
    sht_sp = xls_sp.sheet_by_index(0)
    cols_sp = sht_sp.col_values(0)
    sents = list(set(cols_sp))
    return sents

def seed_chunk_compare(seed,chunk):
    # When the head or tail of a noun phrase in the corpus is equal to seed, we consider that it meets our requirements and its context can be involved in the pattern extraction
    sub_seed = seed.split(" ")
    len_sub_seed = len(sub_seed)
    sub_chunk = chunk.split(" ")
    check_same = 0
    check_result = False
    if len(sub_seed) == len(sub_chunk) == 1:
        if sub_chunk[0].lower() == sub_seed[0].lower():
            check_result = True
    if len(sub_chunk) >= len_sub_seed:
        for sub_i in range(len_sub_seed):
            if (sub_chunk[-sub_i-1]).lower() == (sub_seed[-sub_i-1]).lower():
                check_same += 1
        if check_same == len_sub_seed:
            check_result = True
    check_same = 0
    if len(sub_chunk) >= len_sub_seed:
        for sub_i in range(len_sub_seed):
            if (sub_chunk[sub_i]).lower() == (sub_seed[sub_i]).lower():
                check_same += 1
        if check_same == len_sub_seed:
            check_result = True

    return check_result

def sent_constituent_parsing(sentences,java_path,stanford_parser_path,stanford_model_path,output_path):
    parsing_results = dict()
    for sent in tqdm(sentences):
        if 40<=len(sent)<=500:
            os.environ["JAVAHOME"] = java_path
            scp = StanfordParser(path_to_jar=(stanford_parser_path),
                                 path_to_models_jar=(stanford_model_path))
            result = list(scp.raw_parse(sent))
            parsing_results[(sent.lower())] = str(result[0])
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(json.dumps(parsing_results, indent=2))

    return parsing_results

def subtree_pos_check(subtree):
    # Prevent the extracted NP from being a combination of multiple NPs
    check = True
    limit_word_pos = ["IN","CC","TO","SYM",",","-",".",":"]
    for word_pos in subtree.pos():
        if word_pos[1] in limit_word_pos:
            check = None
            break
    return check

def parse_to_pattern(parsing_results,seeds,tokens):
    bef_pattern = list()
    aft_pattern = list()
    sent_ps = list()
    bef_p_log = dict()
    aft_p_log = dict()

    for sent,parsing in parsing_results.items():
        sent_words = list()
        sent_pos = list()
        parsing = Tree.fromstring(parsing)
        for subtree in parsing.subtrees():
            subtree_label = subtree.label()
            tree_pos = subtree.pos()
            if str(subtree_label) == "ROOT":
                for word_pos in tree_pos:
                    if word_pos[1] != "DT" and word_pos[0] not in chunk_limit:
                        sent_words.append(word_pos[0])
                        sent_pos.append(word_pos[1])
                bef_p_log[" ".join(sent_words)] = list()
                aft_p_log[" ".join(sent_words)] = list()

    for seed in tqdm(seeds):
        for sent,parsing in parsing_results.items():

            sent = sent.lower()
            sent_words = list()
            sent_pos = list()
            parsing = Tree.fromstring(parsing)
            for subtree in parsing.subtrees():
                subtree_pos = list()
                np_words = list()
                subtree_label = subtree.label()
                tree_pos = subtree.pos()
                if str(subtree_label) == "ROOT":
                    for word_pos in tree_pos:
                        if word_pos[1] != "DT" and word_pos[0] not in chunk_limit:
                            sent_words.append(word_pos[0])
                            sent_pos.append(word_pos[1])
                sent_ps.append(" ".join(sent_words))
                check_tree = subtree_pos_check(subtree)
                if str(subtree_label) == "NP" or str(subtree.label()) == "NML":
                    if check_tree:
                        for word_pos in tree_pos:
                            if word_pos[1] != "DT" and word_pos[0] not in chunk_limit:
                                np_words.append((word_pos[0].lower()))
                                subtree_pos.append(word_pos[1])
                        if np_words and len(np_words) > 1:
                            for sub_i in range(len(sent_pos)):
                                if sub_i+len(np_words)-1 < len(sent_words)-1:
                                    check_words = sent_words[sub_i:sub_i+len(np_words)]
                                    if np_words == check_words:
                                        if sent_pos[sub_i + len(np_words)-1] == "NN":
                                            if sent_pos[sub_i + len(np_words)] == "VBG" or sent_pos[sub_i + len(np_words)] == "NN" or sent_pos[sub_i + len(np_words)] == "VBN" or sent_words[sub_i + len(np_words)].endswith("ing") or sent_words[sub_i + len(np_words)].endswith("ed"):
                                                if sent_words[sub_i + len(np_words)] in tokens:
                                                    np_words.append(sent_words[sub_i + len(np_words)])
                                                    break
                        if len(np_words) == 1:

                            for sub_i in range(len(sent_pos)):
                                if sub_i + len(np_words) - 1 < len(sent_words) - 1:
                                    check_words = sent_words[sub_i:sub_i + len(np_words)]
                                    if np_words == check_words:

                                        if sent_pos[sub_i + len(np_words)] == "VBN" or sent_pos[sub_i + len(np_words)] == "VBG" or sent_words[sub_i + len(np_words)].endswith("ed") or sent_words[sub_i + len(np_words)].endswith("ing"):

                                            np_words.append(sent_words[sub_i + len(np_words)])
                                            new_chunk = "-".join(np_words)
                                            if new_chunk in tokens or sent_words[sub_i + len(np_words)] in tokens:

                                                break
                                            else:
                                                np_words.remove(sent_words[sub_i + len(np_words) - 1])
                                        if sent_pos[sub_i + len(np_words)] == "NN":
                                            np_words.append(sent_words[sub_i + len(np_words)])
                        if np_words:
                            chunks = " ".join(np_words)
                            check_same = seed_chunk_compare(seed, chunks)
                            if check_same:
                                len_seed = len(np_words)
                                len_words = len(sent_words)
                                for sub_i in range(len_words):
                                    if sent_words[sub_i] == np_words[0]:
                                        to_seed = sent_words[sub_i:sub_i + len_seed]
                                        to_seed_w = " ".join(to_seed)
                                        if to_seed_w.lower() == seed.lower():
                                            bef_pos = list()
                                            aft_pos = list()
                                            bef_words = list()
                                            aft_words = list()
                                            for index in range(4):
                                                if sub_i - index - 1 >= 0:
                                                    bef_pos.insert(0, sent_pos[sub_i - index - 1])
                                                    bef_words.insert(0, sent_words[sub_i - index - 1])
                                                if sub_i + len_seed - 1 + index + 1 < len_words:
                                                    aft_pos.append(sent_pos[sub_i + len_seed - 1 + index + 1])
                                                    aft_words.append(sent_words[sub_i + len_seed - 1 + index + 1])

                                            if bef_pos:
                                                bef_chunk_str = " ".join(bef_pos)
                                                for b_r, l in bef_rule.items():
                                                    check = re.findall(b_r, bef_chunk_str)
                                                    if len(bef_words) > (l+2):
                                                        if bef_words[-l-1] == "-":
                                                            bef_words_window = " ".join(bef_words[-l-2:])
                                                        else:
                                                            bef_words_window = " ".join(bef_words[-l:])
                                                    else:
                                                        bef_words_window = " ".join(bef_words[-l:])
                                                    aft_pos_words = sent_pos[sub_i+len_seed]
                                                    a_check = re.findall("JJ|VB", aft_pos_words)
                                                    if check and not bef_words_window.endswith(seed) and not a_check:
                                                        sent_p = " ".join(sent_words)
                                                        bef_p_log[sent_p].append(bef_words_window.lower())
                                                        bef_pattern.append(bef_words_window.lower())
                                                        break
                                            if aft_pos:
                                                aft_chunk_str = " ".join(aft_pos)
                                                for a_r, l in aft_rule.items():
                                                    check = re.findall(a_r, aft_chunk_str)
                                                    if len(aft_words) > (l+2):
                                                        if aft_words[l+1] == "-":
                                                            aft_words_window = " ".join(aft_words[:l+2])
                                                        else:
                                                            aft_words_window = " ".join(aft_words[:l])
                                                    else:
                                                        aft_words_window = " ".join(aft_words[:l])
                                                    if check and not aft_words_window.startswith(seed):
                                                        sent_p = " ".join(sent_words)
                                                        aft_pattern.append(aft_words_window.lower())
                                                        aft_p_log[sent_p].append(aft_words_window.lower())
                                                        break
    sent_ps = list(set(sent_ps))
    bef_pattern = list(set(bef_pattern))
    aft_pattern = list(set(aft_pattern))

    return bef_pattern, aft_pattern ,bef_p_log,aft_p_log,sent_ps

def extraction_pattern_score(parsing_results,bef_pattern,aft_pattern,seeds,bef_p_log,aft_p_log,sent_ps):
    bef_pattern_score_fi = dict()
    aft_pattern_score_fi = dict()
    bef_pattern_score_ni = dict()
    aft_pattern_score_ni = dict()
    bef_pattern_score = dict()
    aft_pattern_score = dict()
    # It is used to record the unique chunk information extracted by each pattern
    bef_pattern_log = dict()
    aft_pattern_log = dict()
    for bef_p in bef_pattern:
        bef_pattern_score_fi[bef_p.lower()] = 0
        bef_pattern_score_ni[bef_p.lower()] = 0
        bef_pattern_log[bef_p.lower()] = list()
    for aft_p in aft_pattern:
        aft_pattern_score_fi[aft_p.lower()] = 0
        aft_pattern_score_ni[aft_p.lower()] = 0
        aft_pattern_log[aft_p.lower()] = list()

    bef_sent = bef_p_log.keys()
    aft_sent = aft_p_log.keys()
    all_sents = list()
    all_sents.extend(bef_sent)
    all_sents.extend(aft_sent)

    for sent in sent_ps:
        for bef_p in bef_pattern:
            if bef_p in sent.lower():
                bef_pattern_score_ni[bef_p] += 1
        for aft_p in aft_pattern:
            if aft_p in sent.lower():
                aft_pattern_score_ni[aft_p] += 1
    for sent in sent_ps:
        if sent in bef_p_log.keys():
            b_window = bef_p_log[sent]
            for bef_p in bef_pattern:
                if bef_p in b_window:
                    bef_pattern_score_fi[bef_p.lower()] += 1
        if sent in aft_p_log.keys():
            a_window = aft_p_log[sent]
            for aft_p in aft_pattern:
                if aft_p in a_window:
                    aft_pattern_score_fi[aft_p.lower()] += 1

    for bef_p in bef_pattern:
        r_i = (bef_pattern_score_fi[bef_p.lower()])/(bef_pattern_score_ni[bef_p.lower()])
        score = r_i*np.log2(bef_pattern_score_fi[bef_p.lower()])
        bef_pattern_score[bef_p.lower()] = score

    for aft_p in aft_pattern:
        r_i = (aft_pattern_score_fi[aft_p.lower()]) / (aft_pattern_score_ni[aft_p.lower()])
        score = r_i * np.log2(aft_pattern_score_fi[aft_p.lower()])
        aft_pattern_score[aft_p.lower()] = score

    return bef_pattern_score, aft_pattern_score

from pattern.en import tag

symbols = ["−", "–", "-", "×", "−", '±']
unit_list = ["K min − 1", "A/cm2", "r/min", "mg/min", "mm/s", "g/l", "mJ/m2", "m/min", "mm/rev", "mg/cm2", "°C/min",
             "mL/min", "m/s", "mm/min", "°C·min−1", "s−1",
             "mL", "vol%", "L", "vol.%", "ml",
             "V", "eV", "Mpa", "MPa", "meV", "mA", "GPa", "Gpa", "keV", "W", "Pa", "kW", "F", "nA", "kV", "nA", "kPa",
             "KHz", "HV", "Hz", "Kv", "mbar",
             "kJ", "pJ", "mJ", "J",
             "μm", "cm", "nm", "mm", "m",
             "h", "days", "s−1", "s−1", "ms", "s", "min", "μs", "ns", "weeks", "hours", "minutes", "minute", "hour",
             "ks",
             "g", "g/L", "pct", "kg", "mg", "t",
             "°C", "deg", "K", "℃", "∘C", "C", "°",
             "mN", "kN", "N",
             "mol",
             "cycles"]


def sent_unit_parsing(sent, unit_list, symbols):
    all_unit_chunks = list()
    no_unit_chunks = list()
    get_unit_chunks = list()
    unit_chunk = list()
    if sent:
        sent_labels = tag(sent)
        num_start = None
        num_unit = list()
        for token_i, label in enumerate(sent_labels):
            if label[1] == "SYM" and sent_labels[token_i + 1][1] == "CD":
                num_unit.append(label[0])
                continue
            if label[1] in symbols and sent_labels[token_i + 1][1] == "CD":
                num_unit.append(label[0])
                continue

            if label[1] == "CC" and not re.findall("[a-zA-Z]+", label[0]) and re.findall("^\d+\.?\d*$",
                                                                                         sent_labels[token_i + 1][0]):
                num_unit.append(label[0])
                continue
            num_check = re.findall("^\d+\.?\d*$", label[0])

            if label[1] == "CD" or label[1] == "JJ":
                if num_check and not num_start:
                    num_start = True

                    num_unit.append(label[0])
                    continue
            if num_check and sent_labels[token_i+1][0] in unit_list:
                num_unit.append(label[0])
                num_start = True
                continue
            if num_start:
                unit_exist = 0
                sub_units = label[0].split("/")
                for unit in unit_list:
                    if unit in sub_units:
                        unit_exist += 1
                if label[1] == "CD" or num_check:
                    num_unit.append(label[0])

                elif label[0] in unit_list or unit_exist > 1 or label[0] in symbols:
                    num_unit.append(label[0])

                else:
                    num_start = None
                    chunk_num = " ".join(num_unit)
                    if all(unit not in chunk_num for unit in unit_list):
                        no_unit_chunks.append(chunk_num)
                        unit_chunk.append(chunk_num)
                    if any(unit in chunk_num for unit in unit_list):
                        get_unit_chunks.append(chunk_num)
                        all_unit_chunks.append(chunk_num)
                        unit_chunk.append(chunk_num)
                    num_unit = list()
            if label == sent_labels[-1]:
                if label[1] == "CD" or label[0] in unit_list:
                    chunk_num = " ".join(num_unit)
                    num_start = None
                    if any(unit in chunk_num for unit in unit_list):
                        get_unit_chunks.append(chunk_num)
                        all_unit_chunks.append(chunk_num)
                        unit_chunk.append(chunk_num)

    return all_unit_chunks

def extraction_np_score(func_bp,func_ap,bef_pattern_score, aft_pattern_score,seeds,parsing_results,p_w,tokens,unit_list,symbols):
    all_pattern_score = dict()
    for f_b in func_bp:
        all_pattern_score[f_b] = bef_pattern_score[f_b]
    for f_a in func_ap:
        all_pattern_score[f_a] = aft_pattern_score[f_a]

    extracted_nps = list()
    nps_score = dict()
    np_extracted_pattern = dict()

    all_nps = list()
    units = list()
    for sent,p_r in tqdm(parsing_results.items()):
        sent_words = list()
        sent_pos = list()
        p_r = Tree.fromstring(p_r)
        for subtree in p_r.subtrees():
            np_words = list()
            if str(subtree.label()) == "ROOT":
                tree_pos = subtree.pos()
                for word_pos in tree_pos:
                    if word_pos[1] != "DT" and word_pos[0] not in chunk_limit:
                        sent_words.append(word_pos[0])
                        sent_pos.append(word_pos[1])
            check_tree = subtree_pos_check(subtree)
            if str(subtree.label()) == "NP" and check_tree:
                tree_pos = subtree.pos()
                for word_pos in tree_pos:
                    if word_pos[1] != "DT" and word_pos[0] not in chunk_limit:
                        np_words.append((word_pos[0].lower()))
                if len(np_words) > 1:
                    for sub_i in range(len(sent_pos)):
                        if sub_i + len(np_words) - 1 < len(sent_words) - 1:
                            check_words = sent_words[sub_i:sub_i + len(np_words)]
                            if np_words == check_words:
                                if sent_pos[sub_i + len(np_words)] == "VBG" or sent_pos[sub_i + len(np_words)] == "NN" or sent_pos[sub_i + len(np_words)] == "VBN" or sent_words[sub_i + len(np_words)].endswith("ed") or sent_words[sub_i + len(np_words)].endswith("ing"):
                                    if sent_words[sub_i + len(np_words)] in token_dict:
                                        np_words.append(sent_words[sub_i + len(np_words)])
                                        break

                if len(np_words) == 1:
                    pt_stemmer = pt.PorterStemmer()
                    for sub_i in range(len(sent_pos)):
                        if sub_i + len(np_words) - 1 < len(sent_words) - 1:
                            check_words = sent_words[sub_i:sub_i + len(np_words)]
                            if np_words == check_words:
                                if sent_pos[sub_i + len(np_words)] == "VBG" or sent_pos[sub_i + len(np_words)] == "NN" or sent_pos[sub_i + len(np_words)] == "VBN" or sent_words[sub_i + len(np_words)].endswith("ed") or sent_words[sub_i + len(np_words)].endswith("ing"):
                                    np_words.append(sent_words[sub_i + len(np_words)])
                                    new_chunk = "-".join(np_words)
                                    if new_chunk in tokens or np_words[-1] in tokens:
                                        break
                                    else:
                                        np_words.remove(sent_words[sub_i + len(np_words) - 1])

                if np_words:

                    chunk = " ".join(np_words)
                    all_nps.append(chunk)
                    sentence = " ".join(sent_words)
                    sentence = sentence.lower()
                    bef_aft = sentence.split(chunk.lower())
                    bef_ws = bef_aft[0]
                    aft_ws = bef_aft[1]

                    for b_p in func_bp:
                        if (bef_ws.strip()).endswith(b_p.lower()):
                            extracted_nps.append((chunk.lower()))
                            if (chunk.lower()) not in np_extracted_pattern.keys():
                                np_extracted_pattern[(chunk.lower())] = list()
                                np_extracted_pattern[(chunk.lower())].append(b_p.lower())
                            else:
                                np_extracted_pattern[(chunk.lower())].append(b_p.lower())

                    for a_p in func_ap:
                        if (aft_ws.strip()).startswith(a_p.lower()):
                            extracted_nps.append(chunk.lower())
                            if chunk.lower() not in np_extracted_pattern.keys():
                                np_extracted_pattern[(chunk.lower())] = list()
                                np_extracted_pattern[(chunk.lower())].append(a_p.lower())
                            else:
                                np_extracted_pattern[(chunk.lower())].append(a_p.lower())
        unit = sent_unit_parsing(sent,unit_list,symbols)
        units.extend(unit)
    extracted_nps = list(set(extracted_nps))
    new_extracted_nps = list()
    for e_n in extracted_nps:
        num_check = re.findall("\d+",e_n)
        if num_check:
            if any(e_n.startswith(unit) for unit in units) and any(e_n.endswith(token) for token in tokens):
                new_extracted_nps.append(e_n)
        else:
            new_extracted_nps.append(e_n)

    for np in new_extracted_nps:
        if np not in seeds:
            score_np = 0
            np_patterns = list(set(np_extracted_pattern[np]))
            for n_pt in np_patterns:
                score_np += 1+p_w*all_pattern_score[n_pt]
            nps_score[np] = score_np
    return nps_score

def best_extraction_np(func_bp,func_ap,bef_pattern_score,aft_pattern_score,iteration):
    bef_sents = list()
    bef_scores = list()
    aft_sents = list()
    aft_scores = list()
    for sent,score in bef_pattern_score.items():
        bef_sents.append(sent)
        bef_scores.append(score)
    for sent,score in aft_pattern_score.items():
        aft_sents.append(sent)
        aft_scores.append(score)

    bef_best_ep = dict()
    aft_best_ep = dict()
    all_score = list()
    if iteration == 0:
        bef_m_score = max(bef_scores)
        # if bef_m_score >= aft_m_score:
        for s_i in range(len(bef_scores)):
            if bef_scores[s_i] == bef_m_score and bef_m_score>=0.1:
                func_bp.append(bef_sents[s_i])
                bef_best_ep[bef_sents[s_i]] = bef_scores[s_i]
                print("Before extraction pattern:")
                print(bef_sents[s_i])
        if len(func_bp) <= 2:
            bef_scores_copy = deepcopy(bef_scores)
            bef_sents_copy = deepcopy(bef_sents)

            for s_i in range(len(bef_scores_copy)):
                if bef_sents_copy[s_i] in func_bp:
                    bef_sents.remove(bef_sents[s_i])
                    bef_scores.remove(bef_scores[s_i])
            bef_m_score = max(bef_scores)
            for s_i in range(len(bef_scores)):
                if bef_scores[s_i] == bef_m_score and bef_m_score>=0.1:
                    func_bp.append(bef_sents[s_i])
                    bef_best_ep[bef_sents[s_i]] = bef_scores[s_i]
                    print("Before extraction pattern:")
                    print(bef_sents[s_i])

        aft_m_score = max(aft_scores)
        for s_i in range(len(aft_scores)):
            if aft_scores[s_i] == aft_m_score and aft_m_score>=0.01:
                func_ap.append(aft_sents[s_i])
                aft_best_ep[aft_sents[s_i]] = aft_scores[s_i]
                print("After extraction pattern:")
                print(aft_sents[s_i])
        if len(func_ap) <= 2:
            aft_scores_copy = deepcopy(aft_scores)
            aft_sents_copy = deepcopy(aft_sents)

            for s_i in range(len(aft_scores_copy)):
                if aft_sents_copy[s_i] in func_ap:
                    aft_sents.remove(aft_sents[s_i])
                    aft_scores.remove(aft_scores[s_i])
            aft_m_score = max(aft_scores)
            for s_i in range(len(aft_scores)):
                if aft_scores[s_i] == aft_m_score and aft_m_score>=0.01:
                    func_ap.append(aft_sents[s_i])
                    aft_best_ep[aft_sents[s_i]] = aft_scores[s_i]
                    print("After extraction pattern:")
                    print(aft_sents[s_i])
    else:

        bef_sents_ = list()
        bef_scores_ = list()
        b_sent_socre = dict()
        for sent, score in bef_pattern_score.items():
            if sent not in func_bp:
                bef_sents_.append(sent)
                bef_scores_.append(score)
                b_sent_socre[sent] = score
        a_sent_socre = dict()
        aft_sents_ = list()
        aft_scores_ = list()
        for sent, score in aft_pattern_score.items():
            if sent not in func_ap:
                aft_sents_.append(sent)
                aft_scores_.append(score)
                a_sent_socre[sent] = score
        all_score.extend(bef_scores_)
        all_score.extend(aft_scores_)
        bef_m_score = max(bef_scores_)
        pre_bef_pattern = list()
        for s_i in range(len(bef_scores_)):
            if bef_scores_[s_i] == bef_m_score and bef_sents_[s_i] not in func_bp and bef_m_score>=0.1:
                pre_bef_pattern.append(bef_sents_[s_i])
                print("Before extraction pattern:")
                print(bef_sents_[s_i])
        bef_patterns = pre_bef_pattern
        for pa in bef_patterns:
            func_bp.append(pa)
            bef_best_ep[pa] = bef_m_score

        aft_m_score = max(aft_scores_)
        pre_aft_pattern = list()
        for s_i in range(len(aft_scores_)):
            if aft_scores_[s_i] == aft_m_score and aft_sents_[s_i] not in func_ap and bef_m_score>=0.01:
                pre_aft_pattern.append(aft_sents_[s_i])
                print("After extraction pattern:")
                print(aft_sents_[s_i])

        aft_patterns = pre_aft_pattern
        for pa in aft_patterns:
            func_ap.append(pa)
            aft_best_ep[pa] = bef_m_score
    return bef_best_ep,aft_best_ep,func_bp,func_ap

import nltk.stem.porter as pt

def get_hightest_np(np_score,number,reject_np,tokens):

    recommen = list()
    remove_token = list()
    recom_token = list()
    for iter in range(number):
        nps = list()
        scores = list()
        iter_recom_all = list()
        for np,score in np_score.items():
            check_np = re.findall("\s", np)
            if np not in reject_np:
                if check_np:
                    nps.append(np)
                    scores.append(score)
                else:
                    if np.endswith("tion") or np.endswith("ing"):
                        nps.append(np)
                        scores.append(score)
        if scores:
            max_s = max(scores)
            iteration_recom = list()
            for np_i in range(len(nps)):
                if scores[np_i] == max_s:
                    iteration_recom.append(nps[np_i])
                    iter_recom_all.append(nps[np_i])

            if len(iter_recom_all) < 10:
                recommen.extend(iteration_recom)
                for rec in recommen:
                    if rec in np_score.keys():
                        np_score.pop(rec)
            else:
                recommen.extend(iteration_recom)
                break
        else:
            break
    return recommen,remove_token,recom_token



seeds = seed_read(r"...\start_seeds.xlsx")
sentences = sentences_read(r"...\reorg_sents.xlsx")
java_path = r".../jre1.8.0_321/bin/java.exe"
stanford_parser_path = r".../stanford-parser-full-2020-11-17/stanford-parser.jar"
stanford_model_path = r".../stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar"
output_path = r"...\sent_pos_results.json"

with open(output_path, "r", encoding='utf-8') as f:
    parsing_results = json.load(f)
iterations = 50
func_bp = list()
func_ap = list()
reject_np = list()

with open(r"...\tokens.txt",
        'r', encoding='utf8')as token_dict:
    tokens = eval(token_dict.read())

for iter in range(50):
    # Obtain the extracted pattern for each seed
    bef_pattern,aft_pattern,bef_p_log,aft_p_log,sent_ps = parse_to_pattern(parsing_results,seeds,tokens)
    print("step 1")
    bef_pattern_score,aft_pattern_score = extraction_pattern_score(parsing_results,bef_pattern,aft_pattern,seeds,bef_p_log,aft_p_log,sent_ps)
    print(bef_pattern_score)
    print(aft_pattern_score)
    print("step 2")
    bef_best_ep,aft_best_ep,func_bp,func_ap = best_extraction_np(func_bp,func_ap,bef_pattern_score,aft_pattern_score,iter)
    p_w = 0.01
    print("step 3")
    np_score = extraction_np_score(func_bp,func_ap,bef_pattern_score,aft_pattern_score,seeds,parsing_results,p_w,tokens,unit_list,symbols)
    print("step 4")
    recom_seeds,remove_token,recom_token = get_hightest_np(np_score,15,reject_np,tokens)
    new_recom_seeds = list()
    for r_s in recom_seeds:
        if r_s not in seeds:
            new_recom_seeds.append(r_s)
    new_recom_seeds = list(set(new_recom_seeds))
    new_recom_seeds_copy = copy.deepcopy(new_recom_seeds)
    for np_r in new_recom_seeds_copy:
        if np_r in reject_np:
            new_recom_seeds.remove(np_r)
    r_word = new_recom_seeds
    for np_r in new_recom_seeds:
        if np_r not in r_word:
            reject_np.append(np_r)
    print("This iteration recommed:",r_word)
    with open(r"...\recommended_chunks.txt",encoding="utf-8",mode="a+") as out_f:
        out_f.write("\n")
        out_f.write(str(r_word))
    if r_word:
        seeds.extend(r_word)
    else:
        print("No more process NPs")
        break


