# -*- coding: utf-8 -*-


import xlrd
import re
import json
import nltk
from nltk.tree import Tree
from stanfordcorenlp import StanfordCoreNLP
import copy
from nltk.stem import WordNetLemmatizer
from chemdataextractor.doc import Paragraph
from func_timeout import func_set_timeout,FunctionTimedOut
import configparser as cp

@func_set_timeout(120)
def sent_constituent_parsing(sent, nlp):
    parsing_result = None
    # sent = urllib.parse.quote(sent, encoding='utf8')
    if 40 <= len(sent):
        try:
            parsing_result = nlp.parse(sent)
        except Exception as e:
            print(sent)
            print(e)

    return parsing_result

def chunk_compare(chunk,wnl_chunk_list,wnl_token_dicts,tokens_dict, chunk_dicts):
    lemmatizer = WordNetLemmatizer()
    chunks = nltk.word_tokenize(chunk.lower())
    compare_res = None
    if len(chunks) > 1:
        chunk_tag = nltk.pos_tag(chunks)
        lem_chunk_words = list()
        for t_w_i in range(len(chunk_tag)):
            if chunk_tag[t_w_i][0].startswith("V"):
                lem_word = lemmatizer.lemmatize(chunk_tag[t_w_i][0], 'v')
            else:
                lem_word = lemmatizer.lemmatize(chunk_tag[t_w_i][0], 'n')
            lem_chunk_words.append(lem_word)
        # print(lem_chunk_words)
        if lem_chunk_words:
            wnl_bef = " ".join(lem_chunk_words[:-1])
            all_wnl = " ".join(lem_chunk_words)
            # 与字典中的任一词全匹配或者部分匹配
            for wnl_chunk in wnl_chunk_list:
                if wnl_chunk.endswith(lem_chunk_words[-1]):
                    if any(all_wnl.endswith(wnl_c) for wnl_c in wnl_chunk_list) or any(all_wnl.endswith(wnl_c) for wnl_c in chunk_dicts):
                        # print(all_wnl,"here")
                        compare_res = True
                        # print(chunk, "3")
                        break
            if lem_chunk_words[-1] in tokens_dict:
                compare_res = True
                # print(chunk, "4")
            elif lem_chunk_words[-1] in wnl_token_dicts and chunks[-1].endswith("ing"):
                # print(lem_chunk_words)
                compare_res = True
                # print(chunk, "5")
    return compare_res

def parags_read(parag_path):
    # 读取所有的doi以及对应的段落信息
    lower_chunks = list()
    with open(parag_path, "r", encoding="utf-8") as file:
        parags = json.load(file)

    return parags

def token_candidate_check(sent, token_dict):
    tokens = list()
    # print(sent)
    words = nltk.word_tokenize(sent)
    tagged_sentence = nltk.pos_tag(words)
    # print(tagged_sentence)
    for word in words:
        tags = list()
        words = list()
        recommed_check = None
        for sub_p in tagged_sentence:
            if sub_p[1] != "DT":
                words.append(sub_p[0])
                tags.append(sub_p[1])
        for w_i in range(len(words)):
            if words[w_i] == word:
                nn_check = re.findall("^NN", tags[w_i])
                if tags[w_i] == "VB":
                    aft_words = None
                    aft_tags = None
                    bef_words = words[:w_i]
                    bef_tags = tags[:w_i]
                    if w_i + 1 < len(words):
                        aft_words = words[w_i + 1]
                        aft_tags = tags[w_i + 1]
                    if bef_words and aft_words:
                        b_tags = " ".join(bef_tags)
                        if len(bef_words) >= 3:
                            a_check = re.findall("^NN|^JJ|VB[DN]?$|CD", aft_tags)
                            if not a_check and word in token_dict:  # and any(w_n in bef_sent for w_n in sample_name)
                                recommed_check = True
                if tags[w_i] == "VBD":
                    aft_words = None
                    aft_tags = None
                    bef_words = words[:w_i]
                    bef_tags = tags[:w_i]
                    if w_i + 1 < len(words):
                        aft_words = words[w_i + 1]
                        aft_tags = tags[w_i + 1]
                    if bef_words and aft_words:
                        b_tags = " ".join(bef_tags)
                        # a_tags = " ".join(aft_tags)
                        if len(bef_words) >= 3:
                            # print(word,aft_tags)
                            a_check = re.findall("^NN|^JJ|VB[DN]?$|CD", aft_tags)
                            a_sym_check = re.findall("[a-zA-Z]+", aft_tags)
                            bef_sent = " ".join((bef_words[-3:]))
                            if not a_check and word in token_dict:  # and any(w_n in bef_sent for w_n in sample_name)
                                recommed_check = True
                            # if aft_tags.startswith("IN") and word in token_dict:
                            #     recommed_check = True
                            # if b_in_check and not a_sym_check:
                            #     recommed_check = True

                        elif len(bef_words) == 2:
                            # b_check = re.findall("NN.*\sVB[DNP]$", b_tags)
                            a_check = re.findall("NN|JJ|VB[DN]?$|CD", aft_tags)
                            bef_sent = " ".join((bef_words[-2:]))
                            if not a_check and word in token_dict:
                                # and any(w_n in bef_sent for w_n in sample_name)
                                recommed_check = True
                if tags[w_i] == "VBN":
                    aft_words = None
                    aft_tags = None
                    bef_words = words[:w_i]
                    bef_tags = tags[:w_i]
                    if w_i + 1 < len(words):
                        aft_words = words[w_i + 1]
                        aft_tags = tags[w_i + 1]
                    if bef_words and aft_words:
                        b_tags = " ".join(bef_tags)
                        # a_tags = " ".join(aft_tags)
                        if len(bef_words) >= 3:
                            # print(word,aft_tags)
                            a_check = re.findall("^NN|^JJ|VB[DN]?$", aft_tags)
                            a_sym_check = re.findall("[a-zA-Z]+", aft_tags)
                            bef_sent = " ".join((bef_words[-3:]))
                            if not a_check and word in token_dict:  # and any(w_n in bef_sent for w_n in sample_name)
                                recommed_check = True

                        elif len(bef_words) == 2:
                            # b_check = re.findall("NN.*\sVB[DNP]$", b_tags)
                            a_check = re.findall("NN|JJ|VB[DN]?$", aft_tags)
                            bef_sent = " ".join((bef_words[-2:]))
                            if not a_check and word in token_dict:
                                # and any(w_n in bef_sent for w_n in sample_name)
                                recommed_check = True
                if tags[w_i] == "JJ":
                    if word.endswith("ed") or word.endswith("ing"):
                        aft_words = None
                        aft_tags = None
                        tags_b_words = None
                        bef_words = words[:w_i]
                        bef_tags = tags[:w_i]
                        if w_i + 1 < len(words):
                            aft_words = words[w_i + 1]
                            aft_tags = tags[w_i + 1]
                        if bef_words and aft_words:
                            b_tags = " ".join(bef_tags)
                            # a_tags = " ".join(aft_tags)
                            if len(bef_words) >= 3:
                                # b_check = re.findall("VB[DNP][\sRB]*$", b_tags)
                                # b_and_check = re.findall("VB[DNP]\sCC[\sRB]*$", b_tags)
                                a_check = re.findall("^NN|^JJ|VB[DN]?$|CD", aft_tags)
                                bef_sent = " ".join((bef_words[-3:]))
                                if not a_check and word in token_dict:  # and any(w_n in bef_sent for w_n in sample_name)
                                    recommed_check = True
                            elif len(bef_words) == 2:
                                # b_check = re.findall("NN.*\sVB[DNP]$|PRP\sVB[DNP]$", b_tags)
                                a_check = re.findall("NN|JJ|VB[DN]?$|CD", aft_tags)
                                bef_sent = " ".join((bef_words[-2:]))
                                if not a_check and word in token_dict:
                                    # and any(w_n in bef_sent for w_n in sample_name)
                                    recommed_check = True

                if tags[w_i] == "VBG":  # or tags[w_i] == "JJ"  or nn_check
                    bef_words = None
                    bef_tags_1 = None
                    aft_words = None
                    aft_tags = None
                    tags_b_words = None
                    b_check_2 = None
                    b_check_3 = None
                    a_check_3 = None
                    if w_i - 2 >= 0:
                        bef_words = words[w_i - 2:w_i]
                        bef_tags = tags[w_i - 2:w_i]
                        tags_b_words = " ".join(bef_tags)
                    elif w_i - 1 >= 0:
                        bef_words = words[w_i - 1]
                        bef_tags_1 = tags[w_i - 1]
                    if w_i + 1 < len(words):
                        aft_words = words[w_i + 1]
                        aft_tags = tags[w_i + 1]
                    if bef_words and aft_words:
                        # b_tags = " ".join(bef_tags)
                        # a_tags = " ".join(aft_tags)
                        if tags_b_words:
                            b_check_2 = re.findall("^VB[DNP]\sIN$", tags_b_words)
                            b_check_3 = re.findall("IN$|CC$", tags_b_words)
                        elif bef_tags_1:
                            b_check_1 = re.findall("IN", bef_tags_1)
                        if aft_tags:
                            a_check = re.findall("JJ|VB[DN]?$|CD|NN", aft_tags)
                            a_check_3 = re.findall("IN", aft_tags)
                        if len(bef_words) == 1:

                            if not a_check and word in token_dict:
                                recommed_check = True
                        if len(bef_words) == 2:
                            if not a_check and word in token_dict:
                                recommed_check = True
                            if b_check_3 and a_check_3 and word in token_dict:
                                recommed_check = True
                    if aft_tags:
                        a_check_3 = re.findall("IN", aft_tags)
                    if not bef_words:
                        if a_check_3 and word.lower() in token_dict:
                            # print(word)
                            recommed_check = True
                if nn_check and word.endswith("ing"):
                    bef_words = None
                    bef_tags_1 = None
                    aft_words = None
                    aft_tags = None
                    tags_b_words = None
                    b_check_2 = None
                    if w_i - 2 >= 0:
                        bef_words = words[w_i - 2:w_i]
                        bef_tags = tags[w_i - 2:w_i]
                        tags_b_words = " ".join(bef_tags)
                    elif w_i - 1 >= 0:
                        bef_words = words[w_i - 1]
                        bef_tags_1 = tags[w_i - 1]
                    if w_i + 1 < len(words):
                        aft_words = words[w_i + 1]
                        aft_tags = tags[w_i + 1]
                    aft_all_words = words[w_i:]
                    aft_sent = " ".join(aft_all_words)
                    if bef_words and aft_words:
                        # b_tags = " ".join(bef_tags)
                        # a_tags = " ".join(aft_tags)
                        if tags_b_words:
                            b_check_2 = re.findall("^VB[DN]\sIN$|.*CC$|.*,$", tags_b_words)
                        elif bef_tags_1:
                            b_check_1 = re.findall("IN", bef_tags_1)
                        if aft_tags:
                            a_check = re.findall("JJ|VB[DN]$|CD|NN", aft_tags)
                        if len(bef_words) == 1:
                            if not a_check and b_check_1 and word in token_dict:
                                recommed_check = True
                        if len(bef_words) == 2:
                            if b_check_2 and word in token_dict:
                                recommed_check = True
        if recommed_check:
            for w_i, wo in enumerate(words):
                if word == wo:
                    phrase = list()
                    phrase.append(word)
                    tags_c = tags[:w_i]
                    words_c = words[:w_i]
                    tags_c.reverse()
                    words_c.reverse()
                    for c_i in range(len(tags_c)):
                        if tags_c[c_i] == "RB" or tags_c[c_i] == "JJ":
                            phrase.insert(0, words_c[c_i])
                        else:
                            break
                    # print(phrase)
                    word_out = " ".join(phrase)
                    # print(word)
                    tokens.append(word_out)
    tokens = list(set(tokens))
    return tokens

def wnl_tokens(token_dict):
    all_wnls = list()
    lemmatizer = WordNetLemmatizer()
    for e_c in token_dict:
        lem_word = lemmatizer.lemmatize(e_c, 'v')
        all_wnls.append(lem_word)
        all_wnls.append(e_c)
    all_wnls = list(set(all_wnls))
    return all_wnls

def subtree_pos_check(subtree):
    # 防止抽取的到的NP是由多个NP的组合体
    check = True
    limit_word_pos = ["IN", "CC", "TO", "SYM", ",", "-", ".", ":"]
    for word_pos in subtree.pos():
        if word_pos[1] in limit_word_pos:
            check = None
            break

    return check

def chunk_candidated(sent, token_dict, chunk_dict, nlp,unit_list):
    all_nps = list()
    p_r = None
    limit_pos = ["IN", "TO", "SYM", ",", "-", ".", ":"]#存在limit_pos即为非纯粹的chunk
    try:
        p_r = sent_constituent_parsing(sent,nlp)
    except FunctionTimedOut as e:
        print('function timeout + msg = ', e.msg)
        # print(sent)
    sent_words = list()
    sent_pos = list()

    if p_r:
        p_r = Tree.fromstring(p_r)
        np_bef = ""
        for subtree in p_r.subtrees():
            np_words = list()
            nn_nps = list()
            if str(subtree.label()) == "ROOT":
                tree_pos = subtree.pos()
                for word_pos in tree_pos:
                    if word_pos[1] != "DT" and word_pos[0] not in unit_list:#and word_pos[1] != "CD"
                        sent_words.append(word_pos[0])
                        sent_pos.append(word_pos[1])
            # print(sent_words)
            # print(sent_pos)
            tree_check = None
            check_tree = subtree_pos_check(subtree)
            # 类似 water quenching在被解析时，会单独针对water给出一个NP的subtree
            if str(subtree.label()).startswith("N"):
                if len(subtree.pos())>1:

                    if not tree_check and not check_tree and ('and', 'CC') in subtree.pos() and all(sub[1] not in limit_pos for sub in subtree.pos()):
                        sub_words = list()
                        for word_pos in subtree.pos():
                            if word_pos[1] != "CC" and word_pos[1] != "CD": # and word_pos[0] not in unit_list and word_pos[1] != "CD"
                                sub_words.append((word_pos[0]))
                            else:
                                if len(sub_words)>1:
                                    entity = " ".join(sub_words)
                                    all_nps.append(entity)
                                sub_words = list()
                        if len(sub_words)>1:
                            entity = " ".join(sub_words)
                            all_nps.append(entity)

                    if not tree_check and check_tree:
                        # 只拿取所有长度超过1的词块
                        # if subtree.pos()[-1][0] not in unit_list:
                        for word_pos in subtree.pos():
                            if word_pos[1] != "DT": # and word_pos[0] not in unit_list and word_pos[1] != "CD"
                                np_words.append((word_pos[0]))
                        np_c = " ".join(np_words)
                        if np_c not in np_bef:
                            np_bef = np_c
                            for itera in range(5):
                                if np_words and len(np_words) > 1:
                                    for sub_i in range(len(sent_pos)):
                                        # check_sub = sent_p[sub_i:sub_i+len_tree]
                                        # check_words = list()
                                        if sub_i+len(np_words)-1 < len(sent_words)-1:
                                            check_words = sent_words[sub_i:sub_i+len(np_words)]
                                            if np_words == check_words:

                                                if sent_pos[sub_i + len(np_words)] == "HYPH" or sent_pos[sub_i + len(np_words)] == "VBG" or sent_pos[sub_i + len(np_words)] == "NN" or sent_pos[sub_i + len(np_words)] == "VBN" or sent_words[sub_i + len(np_words)].endswith("ed") or sent_words[sub_i + len(np_words)].endswith("ing"):
                                                    if sent_words[sub_i + len(np_words)] in token_dict:
                                                        # print(sent_words[sub_i + len(np_words)])
                                                        np_words.append(sent_words[sub_i + len(np_words)])
                                                        break

                                if len(np_words) == 1:#and np_words[0] in chunk_dict
                                    # print(np_words)
                                    for sub_i in range(len(sent_pos)):
                                        if sub_i+len(np_words)-1 < len(sent_words)-1:
                                            check_words = sent_words[sub_i:sub_i+len(np_words)]
                                            if np_words == check_words:
                                                    if sent_pos[sub_i + len(np_words)] == "VBG" or sent_pos[sub_i + len(np_words)] == "VBN" or sent_words[sub_i + len(np_words)].endswith("ed") or sent_words[sub_i + len(np_words)].endswith("ing"):

                                                        np_words.append(sent_words[sub_i + len(np_words)])

                                                        new_chunk = "-".join(np_words)
                                                        chunk = " ".join(np_words)
                                                        if new_chunk in token_dict or np_words[-1] in token_dict or chunk in chunk_dict:
                                                            # print("333", chunk)
                                                            all_nps.append(chunk)
                                                        else:
                                                            np_words.remove(sent_words[sub_i + len(np_words)-1])
                                                    if sent_pos[sub_i + len(np_words)] == "NN" or sent_pos[sub_i + len(np_words)] == "HYPH":
                                                        np_words.append(sent_words[sub_i + len(np_words)])
                                                    elif sent_pos[sub_i + len(np_words)] == "VB":
                                                        np_words.append(sent_words[sub_i + len(np_words)])
                            if np_words and len(np_words) > 1:
                                chunk = " ".join(np_words)
                                all_nps.append(chunk)

                else:
                    np_words.append(subtree.pos()[0][0])
                    for itera in range(3):# 词性解析在材料尝试往后继续寻找相关的词
                        for sub_i in range(len(sent_pos)):
                            if sub_i + len(np_words) - 1 < len(sent_words) - 1:
                                check_words = sent_words[sub_i:sub_i + len(np_words)]
                                if np_words == check_words:
                                    if sent_pos[sub_i + len(np_words)] == "VB" or sent_pos[sub_i + len(np_words)] == "VBG" or sent_pos[sub_i + len(np_words)] == "VBN" or sent_words[sub_i + len(np_words)].endswith("ed") or sent_words[sub_i + len(np_words)].endswith("ing"):
                                        np_words.append(sent_words[sub_i + len(np_words)])
                                        new_chunk = "-".join(np_words)
                                        chunk = " ".join(np_words)
                                        if new_chunk in token_dict or chunk in chunk_dict: # np_words[-1] in token_dict or
                                            nn_nps.append(chunk)
                                        else:
                                            np_words.remove(sent_words[sub_i + len(np_words) - 1])
                                    elif sent_pos[sub_i + len(np_words)] == "NN" or sent_pos[sub_i + len(np_words)] == "HYPH":
                                        np_words.append(sent_words[sub_i + len(np_words)])
                    if np_words and len(np_words) > 1:
                        chunk = " ".join(np_words)

                        nn_nps.append(chunk)
            for nn in nn_nps:
                if all(nn not in pos_n for pos_n in all_nps):
                    all_nps.append(nn)
    # print(all_nps,"111")
    all_nps = list(set(all_nps))
    return all_nps

def wnl_chunks(chunk_dict):
    lemmatizer = WordNetLemmatizer()
    all_wnls = list()
    for e_c in chunk_dict:
        e_c = e_c.replace("-lrb-", "(")
        e_c = e_c.replace("-rrb-", ")")
        e_c = re.sub("\s*\(.+\)","",e_c)
        lem_e_chunk_words = list()
        e_cs = nltk.word_tokenize(e_c)
        e_chunk_tag = nltk.pos_tag(e_cs)
        for e_w in e_chunk_tag:

            if e_w[1].startswith("V"):
                lem_word = lemmatizer.lemmatize(e_w[0], 'v')
            else:
                lem_word = lemmatizer.lemmatize(e_w[0], 'n')
            lem_e_chunk_words.append(lem_word)
        wnl = " ".join(lem_e_chunk_words)
        if wnl:
            all_wnls.append(wnl)
    return all_wnls

def get_candidated_action(sent,token_dict,chunk_dict,wnl_chunk_dicts,wnl_token_dicts,nlp,unit_list):
    sent_in_tokens = list()
    sent_in_chunks = list()
    # print(sent)
    chunks = chunk_candidated(sent,token_dict,chunk_dict, nlp,unit_list)
    extra_chunks = list()
    tokens = token_candidate_check(sent,token_dict)
    for token in tokens:
        tokens = nltk.word_tokenize(token)
        if (tokens[-1].lower()) in token_dict and token not in sent_in_tokens:
            sent_in_tokens.append(token)
    for chunk in chunks:
        chunk = chunk.replace("-lrb-", "(")
        chunk = chunk.replace("-rrb-", ")")
        chunk = chunk.replace("-LRB-", "(")
        chunk = chunk.replace("-RRB-", ")")
        chunk = re.sub("\s*\(.+\)","",chunk)
        if chunk:
            chunk_check = chunk_compare(chunk, wnl_chunk_dicts,wnl_token_dicts,token_dict,chunk_dict)
            num_c = re.findall("\d+", chunk)
            if chunk_check:
                sent_in_chunks.append(chunk)
            elif not num_c:
                extra_chunks.append(chunk)
    return sent_in_tokens,list(set(sent_in_chunks)),list(set(extra_chunks))

def subtree_pos_check(subtree):
    # 防止抽取的到的NP是由多个NP的组合体
    check = True
    limit_word_pos = ["IN", "CC", "TO", "SYM", ",", "-", ".", ":"]
    for word_pos in subtree.pos():
        if word_pos[1] in limit_word_pos:
            check = None
            break
    return check

def para_add(sent, unit_list):
    for unit in unit_list:
        for unit_ in unit_list:
            if unit + " / " + unit_ in sent:
                sent = sent.replace(unit + " / " + unit_, unit + "/" + unit_)
    param_0 = re.findall("(\s+\d+\.?\d*\s*\,)+\s*\d*\.?\d*\s+and\s+\d+\.?\d*\s+", sent)
    param_2 = re.findall("\s+\d+\.?\d*\s+to\s+\d+\.?\d*\s+", sent)

    sent_r = None
    repla_unit = None
    if param_0:
        param_1 = re.findall("[\s\d\.\,]+\s*\d*\.?\d*\s+and\s+\d+\.?\d*\s+", sent)
        # print(param_1)
        params_part = param_1[0]
        two_parts = sent.split(param_1[0])
        unit_get = None
        for unit in unit_list:
            if two_parts[1].startswith(unit):
                # print(sent)
                repla_unit = unit
                unit_get = True
                break
        if not unit_get:
            print("Need to add a unit:")
            print(two_parts[1].split(" ")[0])
        if unit_get:
            nums = re.findall("\d+\.?\d*", param_1[0])
            params_part_new = copy.deepcopy(params_part)
            # print(params_part_new)
            # print(sent)
            # print(nums[:-1])
            for num in nums[:-1]:
                params_part_new = params_part_new.replace(" " + num + " ", " " + num + " " + repla_unit + " ")
            sent_r = sent.replace(params_part, params_part_new)
    elif param_2:

        two_parts = sent.split(param_2[0])
        params_part = param_2[0]
        unit_get = None
        for unit in unit_list:
            if (two_parts[1]).startswith(unit):
                # print(sent)
                # print(two_parts)

                repla_unit = unit
                # print(str(unit))
                unit_get = True
                break
        if not unit_get:
            print("Need to add a unit:")
            print(two_parts[1].split(" ")[0])
        if unit_get:
            nums = re.findall("\d+\.?\d*", param_2[0])
            params_part_new = copy.deepcopy(params_part)
            # print(params_part_new)
            # print(sent)
            for num in nums[:-1]:
                params_part_new = params_part_new.replace(" " + num + " ", " " + num + " " + repla_unit + " ")
            # print(params_part_new)
            sent_r = sent.replace(params_part, params_part_new)
    else:
        sent_r = sent

    return sent_r

def corpus_preprocess(sent, items, unit_list):
    for item in items:
        sent = sent.replace(item[0], item[1])
    for unit in unit_list:
        unit_check = re.findall("[0-9]\s+\-\s+" + unit, sent)
        space_check = re.findall("\s+", sent)

        if unit_check:
            # print(unit)
            rep = unit_check[0].replace("- ", "")
            sent = sent.replace(unit_check[0], rep)
        # print(sent)
        for sp in space_check:
            sent = sent.replace(sp, " ")
    sent = " ".join(nltk.word_tokenize(sent))
    return sent

def unit_part(r_unit_chunk, unit_list):
    chunk_part = r_unit_chunk.split(" ")
    bef_num = None
    unit_part = None
    unit = None
    if chunk_part[-1] in unit_list:
        unit = chunk_part[-1]
    return unit

def unit_process(unit, unit_list):
    unit_space = re.sub("\s+", " ", unit)
    unit_parts = unit_space.split(" ")
    new_unit = None
    if len(unit_parts) == 3:
        if unit_parts[0] in unit_list and unit_parts[1] in unit_list:
            num = re.findall("[\−\−\-]{1}(\d+)", unit_parts[2])
            if num:
                new_unit = unit_parts[0] + "/" + unit_parts[1] + str(num)
    return new_unit

def link_compare(sent, no_unit_chunk, r_units_chunks, unit_list):
    # print(r_units_chunks)
    together_rela = None
    unit = None
    new_chunk = None
    new_unit = None
    new_chunk_output = None
    true_words = [",", "and", "to", ", and"]
    for r_unit_chunk in r_units_chunks:
        unit = unit_part(r_unit_chunk, unit_list)
        if unit:
            # print(unit)
            new_unit = unit_process(unit, unit_list)
            if new_unit:
                # print(new_unit)
                new_chunk = r_unit_chunk.replace(unit, new_unit)
                sent = sent.replace(r_unit_chunk, new_chunk)
                two_parts = sent.split(new_chunk)
                for part in two_parts:
                    if " " + no_unit_chunk + " " in part:
                        sub_parts = part.split(" " + no_unit_chunk + " ")
                        check_part = sub_parts[1]
                        # print(check_part)
                        for t_w in true_words:
                            if check_part.strip == t_w:
                                together_rela = True
                                new_chunk_output = no_unit_chunk + " " + new_unit + " "
                                rep_chunk = no_unit_chunk + check_part + r_unit_chunk

                                sent = sent.replace(no_unit_chunk + " " + check_part,
                                                    no_unit_chunk + " " + new_unit + " " + check_part)
                                break
                                # elif unit:
                                #     sent = sent.replace(no_unit_chunk+check_part+r_unit_chunk, no_unit_chunk+" "+unit+" "+check_part+r_unit_chunk)
            else:
                two_parts = sent.split(r_unit_chunk)
                # if r_unit_chunk == "221.5 J":
                #             print(check_part)
                for part in two_parts:
                    if " " + no_unit_chunk + " " in part:
                        sub_parts = part.split(" " + no_unit_chunk + " ")
                        check_part = sub_parts[1]
                        # if r_unit_chunk == "221.5 J ":
                        # print(check_part)
                        for t_w in true_words:
                            if check_part.strip() == t_w:
                                together_rela = True
                                new_chunk_output = no_unit_chunk + " " + unit
                                sent = sent.replace(no_unit_chunk + " " + check_part,
                                                    no_unit_chunk + " " + unit + " " + check_part)
                                break
    return together_rela, sent, new_chunk_output

def sent_unit_parsing(sent, replace_items, unit_list,symbols):
    all_unit_chunks = list()
    no_unit_chunks = list()
    get_unit_chunks = list()
    unit_chunk = list()
    sent = corpus_preprocess(sent, replace_items, unit_list)
    # sent = para_add(sent,unit_list)
    if sent:
        sent_words = nltk.word_tokenize(sent)
        sent_labels = nltk.pos_tag(sent_words)
        num_start = None
        num_unit = list()
        for token_i, label in enumerate(sent_labels):
            # if token_i < len(sent_labels)-1:
            if label[1] == "SYM" and sent_labels[token_i + 1][1] == "CD":
                num_unit.append(label[0])
                continue
            if label[1] in symbols and sent_labels[token_i + 1][1] == "CD":
                num_unit.append(label[0])
                continue

            if label[1] == "CC" and not re.findall("[a-zA-Z]+", label[0]) and re.findall("^[–-−]?\d+[,.–-−×±]*\d*",
                                                                                         sent_labels[token_i + 1][0]):
                num_unit.append(label[0])
                continue
            num_check = re.findall("^[–-−]?\d+[,.–-−×±]*\d*$", label[0])#"^\d+\,?\d*\.?\d*$"
            num = re.findall("[–-−]?\d+[,.–-−×±]*\d*", label[0])

            if label[1]=="CD" and num:
                if any(label[0]==num[0]+unit for unit in unit_list):
                    if not num_start:
                        num_start = True
                    num_unit.append(label[0])
                    continue

            if label[1] == "CD" or label[1] == "JJ":

                if num_check and not num_start:

                    num_start = True
                    num_unit.append(label[0])
                    continue
            if num_check and token_i + 1 < len(sent_labels):
                if sent_labels[token_i + 1][0] in unit_list:
                    num_unit.append(label[0])
                    num_start = True
                    continue
            # print(num_unit)
            if num_start:
                unit_exist = 0
                sub_units = label[0].split("/")
                for unit in unit_list:
                    if unit in sub_units:
                        # print(unit)
                        # print(label[0])
                        unit_exist += 1
                if label[1] == "CD" or label[1] == "$" or num_check:
                    num_unit.append(label[0])
                    # print(label)
                elif label[0] in unit_list or unit_exist >= 1 or label[0] in symbols:

                    num_unit.append(label[0])
                else:
                    num_start = None
                    # if len(num_unit) == 2:
                    chunk_num = " ".join(num_unit)
                    # if chunk_num == "300":
                    #     print(sent)

                    if all(unit not in chunk_num for unit in unit_list):
                        no_unit_chunks.append(chunk_num)
                        unit_chunk.append(chunk_num)
                        # print(chunk_num)
                        # print(sent)
                    if any(unit in chunk_num for unit in unit_list):
                        get_unit_chunks.append(chunk_num)
                        all_unit_chunks.append(chunk_num)
                        unit_chunk.append(chunk_num)
                    num_unit = list()
            if token_i == len(sent_labels)-1:
            # if label == sent_labels[-1]:
                if label[1] == "CD" or label[0] in unit_list:
                    chunk_num = " ".join(num_unit)
                    num_start = None
                    if any(unit in chunk_num for unit in unit_list):
                        get_unit_chunks.append(chunk_num)
                        all_unit_chunks.append(chunk_num)
                        unit_chunk.append(chunk_num)

        for itera in range(6):
            for no_unit_chunk in no_unit_chunks:
                together_rela, sent, new_chunk_output = link_compare(sent, no_unit_chunk, get_unit_chunks, unit_list)
                # print(together_rela, sent, new_chunk_output)
                if together_rela:
                    all_unit_chunks.append(new_chunk_output)
                    get_unit_chunks.append(new_chunk_output)
                    no_unit_chunks.remove(no_unit_chunk)
                    break
    unit_id = dict()
    new_chunks = list()
    # print(all_unit_chunks,sent)
    # 重新将unit按照它在句中的顺序排序，由于下面要根据顺序索引将unit的label进行替换
    index_log = dict()
    for unit in all_unit_chunks:
        if unit in sent:
            if unit not in index_log.keys():
                unit_id[sent.index(unit)] = unit
                index_log[unit] = list()
                index_log[unit].append(sent.index(unit))
            else:
                start_index = index_log[unit][-1]
                unit_id[sent.index(unit,start_index+1,len(sent))] = unit
                index_log[unit] = list()
                index_log[unit].append(sent.index(unit))
        else:
            pass
            # print(sent,unit)
    unit_id = sorted(unit_id.items(), key=lambda x: x[0])
    for unit in unit_id:
        new_chunks.append(unit[1])
    # print(new_chunks)
    # 解决参数与参数之间的关系
    sent_search = copy.deepcopy(sent)
    replace_sym = list()
    id = 1
    words = nltk.word_tokenize(sent_search)
    sent_search = " ".join(words)
    sent_search = " "+sent_search+" "
    sent_search_n = copy.deepcopy(sent_search)
    for par_unit in new_chunks:
        if " "+par_unit+" " in sent_search_n:
            sent_search_n = sent_search_n.replace(" "+par_unit+" ", " "+"UNIT"+str(id)+" ",1)
            replace_sym.append("UNIT"+str(id))
            id += 1
    # print(sent_search_n)
    all_unit_chunks_ = copy.deepcopy(new_chunks)
    new_unit_chunks = list()
    unit_i = 0
    words = nltk.word_tokenize(sent_search_n)
    unit_relas = list()
    dele_unit = list()
    # 找出符合 word_1 + unit_n + word_2 + unit_n+1且这两unit单位相同的短语保存到unit_relas中
    for unit in replace_sym[:-1]:
        if words.index(unit)+2 < len(words):
            if words[words.index(unit)+2] in replace_sym and words[words.index(unit)+2]==replace_sym[unit_i+1]:
                # print(unit)
                if new_chunks[unit_i][-1]==new_chunks[unit_i+1][-1]:
                    # all_unit_chunks.remove(all_unit_chunks[replace_sym.index(unit)])
                    # all_unit_chunks.remove(all_unit_chunks[replace_sym.index(words[words.index(unit)+2])])
                    mid_w = words[words.index(unit)+1]
                    # print(mid_w)
                    check = re.findall("[a-zA-Z]",mid_w)
                    if check:
                        phrase = words[words.index(unit)-1:words.index(unit)+3]
                        unit_phrase = " ".join(phrase)
                        # print(unit_phrase)
                        for rep in replace_sym:
                            if rep in unit_phrase:
                                unit_phrase = unit_phrase.replace(rep,all_unit_chunks_[replace_sym.index(rep)])
                        # print(unit_phrase)
                        unit_phrase = re.sub('[,;]+', "", unit_phrase)
                        unit_relas.append(unit_phrase)
                        dele_unit.append(new_chunks[unit_i+1])
                        if unit_i >= len(replace_sym)-1:
                            break
        unit_i += 1

    return new_chunks, sent, unit_relas

def shortest_match(conj_words, conjs_bef):
    # print(conj_words,conjs_bef)
    word_out = None
    dist_words = list()
    in_verbs = list()
    for word in conj_words:
        if " " + word + " " in conjs_bef:
            dis = conjs_bef.split(" " + word + " ")[-1]
            in_verbs.append(word)
            dist_words.append(len(dis))
    if dist_words:
        min_distance = min(dist_words)
        word_out = in_verbs[dist_words.index(min_distance)]
    return word_out

def sample_sym_process(sentence, verbs, symbols):
    """
    将带有特殊符号、且这些符号在依存解析过程中会将整个词截断分析的词统计出来，并且对它们进行替换，防止它们干扰依存解析，其中被识别为verb的词不进行替换
    :param sentence:
    :param verbs:
    :return:
    """
    rep_dict = dict()
    sym_rule = "\s+[\–\-]\s+"
    new_sentence = re.sub(sym_rule, "", sentence)
    words = nltk.word_tokenize(new_sentence)
    mat_name_rule = ['^[0-9]+\.?[0-9]{0,2}[A-JL-Z]',
                     '[0-9]{0,2}\.?[0-9]{0,2}[A-Z][a-z]?\-[0-9]{0,2}\.?[0-9]{0,2}[A-Z][a-z]?',
                     '^[A-Z]+[a-z]*\-[0-9]\w*', '^[A-Z]\S+[0-9]$', '^[A-Z]+[0-9]+[A-z]+', '^[A-Z]+[a-z]*\~[A-Z0-9]+']

    num = 0
    for word in words:
        name_check = None
        for rule in mat_name_rule:
            check = re.findall(rule, word)
            if check:
                name_check = True
                break
        if any(sym in word for sym in symbols) and name_check:
            if word not in verbs:
                if word not in rep_dict.keys():
                    label = str(num) + "SN"
                    new_sentence = new_sentence.replace(word, label)
                    new_sentence = re.sub(label + "\\s+alloy", label, new_sentence)
                    new_sentence = re.sub("alloy\\s+" + label, label, new_sentence)
                    rep_dict[label] = word
                    num += 1
                else:
                    pass
        for rule in mat_name_rule:
            name_check = re.findall(rule, word)
            if name_check:
                new_sentence = new_sentence.replace(word+" alloy", word)
                new_sentence = new_sentence.replace("alloy " + word, word)
                # new_sentence = re.sub(word + "\s+alloy", word, new_sentence)
                # new_sentence = re.sub("alloy\s+" + word, word, new_sentence)
                break

    return new_sentence, rep_dict

@func_set_timeout(120)
def nlp_dp(nlp,sent):
    dependencyParse = nlp.dependency_parse(sent)
    return dependencyParse

def dp_parsing(tokens_dict,chunks_dict,nlp, sentence, verbs, chunks, units, other_chunks,symbols):
    # 若参数和单位前方存在已经识别到的token或者chunk，那么不管依存解析的结果
    # 若参数和单位前方没有识别到的方法和动作，那么按照依存解析的结果给出这个参数所依赖的值
    # print(chunks,333)
    # print(sentence)
    sentence = " ".join(nltk.word_tokenize(sentence))
    sentence += " "
    sentence = " "+ sentence
    # print(sentence)
    new_verbs = copy.deepcopy(verbs)
    for verb in verbs:
        space_c = re.findall("\s+",verb)
        if space_c:
            chunks.append(verb)
            new_verbs.remove(verb)
    verbs = new_verbs

    sentence, rep_dict = sample_sym_process(sentence, verbs,symbols)
    # print(sentence)
    # print(units)
    # print(chunks)
    ori_sentence = copy.deepcopy(sentence)
    verb_dict = dict()
    phrase_dict = dict()
    act_labels_all = list()
    for phrase_i in range(len(chunks)):
        if chunks[phrase_i].endswith("ed"):
            label = "CN" + str(phrase_i) + "ed"
        else:
            label = "CN" + str(phrase_i)
        if " " + chunks[phrase_i] + " " in sentence:
            sentence = sentence.replace(" " + chunks[phrase_i] + " ", " " + label + " ")
            phrase_dict[label] = chunks[phrase_i]
            act_labels_all.append(label)
            verb_dict[chunks[phrase_i]] = [[],[],[]]
    # print(verb_dict)
    for phrase_i in range(len(chunks), len(chunks) + len(other_chunks)):
        if other_chunks[phrase_i - len(chunks)].endswith("ed"):
            label = "CN" + str(phrase_i) + "ed"
        else:
            label = "CN" + str(phrase_i)
        # print(other_chunks[phrase_i-len(chunks)])
        if " " + other_chunks[phrase_i - len(chunks)] + " " in sentence:
            sentence = sentence.replace(" " + other_chunks[phrase_i - len(chunks)] + " ", " " + label + " ")
            phrase_dict[label] = other_chunks[phrase_i - len(chunks)]
            act_labels_all.append(label)
            verb_dict[other_chunks[phrase_i - len(chunks)]] = [[], [], []]

    verb_sym_dict = dict()

    unit_labels_all = list()
    for verb_i in range(len(verbs)):
        verb_check = re.findall("^[A-Za-z]+$",verbs[verb_i])
        if verb_check:
            if not verbs[verb_i].endswith("ing"):
                label = str(verb_i) + "Ved"
                # print(label)
            else:
                label = str(verb_i) + "Ving"
            if " "+verbs[verb_i]+" " in sentence:
                sentence = sentence.replace(" "+verbs[verb_i]+" ", " "+label+" ")
                verb_sym_dict[label] = verbs[verb_i]
                act_labels_all.append(label)
                verb_dict[verbs[verb_i]] = [[],[],[]]
        elif not verb_check and verbs[verb_i].endswith("ing"):
            label = str(verb_i) + "Ving"
            if " " + verbs[verb_i] + " " in sentence:
                sentence = sentence.replace(" " + verbs[verb_i] + " ", " " + label + " ")
                verb_sym_dict[label] = verbs[verb_i]
                act_labels_all.append(label)
                verb_dict[verbs[verb_i]] = [[], [], []]
        elif not verb_check and verbs[verb_i].endswith("ed"):
            label = str(verb_i) + "Ved"
            if " " + verbs[verb_i] + " " in sentence:
                sentence = sentence.replace(" " + verbs[verb_i] + " ", " " + label + " ")
                verb_sym_dict[label] = verbs[verb_i]
                act_labels_all.append(label)
                verb_dict[verbs[verb_i]] = [[], [], []]
    # print(verbs)
    # print(sentence)
    # print(verb_dict.keys())
    unit_dict = dict()
    for unit_i in range(len(units)):
        label = "UP" + str(unit_i)
        sentence = sentence.replace(" "+units[unit_i]+" ", " "+label+" ")
        # print(label)
        unit_dict[label] = units[unit_i]
        unit_labels_all.append(label)

    token = nlp.word_tokenize(sentence)
    dependencyParse =None
    try:
        dependencyParse = nlp_dp(nlp,sentence)
    except FunctionTimedOut as e:
        print('function timeout + msg = ', e.msg)
    order_verb_dict = None
    if dependencyParse:
        parse_results = list()
        for i, begin, end in dependencyParse:
            one_r = list()
            one_r.append(token[begin - 1])
            one_r.append(i)
            one_r.append(token[end - 1])
            parse_results.append(one_r)

        all_verb_chunk = list()
        all_verb_chunk.extend(verbs)
        all_verb_chunk.extend(chunks)
        copy_units = copy.deepcopy(units)

        # print(verb_dict)
        paren_part = re.findall("\(.+\)",sentence)
        para_unit_part = list()
        for sub_paren_part in paren_part:
            if any(lab in sub_paren_part for lab in unit_labels_all):
                para_unit_part.append(sub_paren_part)
        # print(para_unit_part)
        for lab in act_labels_all:
            # print(lab)
            # print(sentence.split(lab))
            parts = sentence.split(lab)
            for part_i in range(1,len(parts)):
                # if parts[part_i].strip().endswith(lab):
                    for pup in para_unit_part:
                #         if part_i+1 < len(parts):
                        strip_p = parts[part_i].strip()
                        # print(strip_p)
                        if strip_p.startswith(pup):
                            for lab_u in unit_labels_all:
                                if lab_u in pup:
                                    # print(lab)
                                    if lab in verb_sym_dict.keys():
                                        verb_dict[verb_sym_dict[lab]][2].append(unit_dict[lab_u])
                                        if unit_dict[lab_u] in copy_units:
                                            copy_units.remove(unit_dict[lab_u])
                                    elif lab in phrase_dict.keys():
                                        verb_dict[phrase_dict[lab]][2].append(unit_dict[lab_u])
                                        if unit_dict[lab_u] in copy_units:
                                            copy_units.remove(unit_dict[lab_u])

        for p_r in parse_results:
            if p_r[0] in verb_sym_dict.keys():
                if "nsubj" in p_r[1]:
                    if p_r[2] in phrase_dict.keys():
                        if p_r[2] in rep_dict.keys():
                            verb_dict[verb_sym_dict[p_r[0]]][0].append(phrase_dict[rep_dict[p_r[2]]])
                        else:
                            verb_dict[verb_sym_dict[p_r[0]]][0].append(phrase_dict[p_r[2]])
                    elif p_r[2] in rep_dict.keys():
                        verb_dict[verb_sym_dict[p_r[0]]][0].append(rep_dict[p_r[2]])
                    else:
                        verb_dict[verb_sym_dict[p_r[0]]][0].append(p_r[2])
                if "obl" in p_r[1]:
                    unit_chunk = re.findall("UP\d+$", p_r[2])
                    if unit_chunk:
                        # print(p_r[0],p_r[2])
                        verb_dict[verb_sym_dict[p_r[0]]][2].append(unit_dict[p_r[2]])
                        # print(p_r[0],unit_dict[p_r[2]])
                        if unit_dict[p_r[2]] in copy_units:
                            copy_units.remove(unit_dict[p_r[2]])
                if "nmod" in p_r[1]:
                    unit_chunk = re.findall("UP\d+$", p_r[2])
                    if unit_chunk:
                        verb_dict[verb_sym_dict[p_r[0]]][2].append(unit_dict[p_r[2]])
                        # print(p_r[0],unit_dict[p_r[2]])
                        if unit_dict[p_r[2]] in copy_units:
                            copy_units.remove(unit_dict[p_r[2]])
                if "conj" in p_r[1]:
                    if p_r[2] in verbs:
                        # print(p_r[2],"111")
                        verb_dict[verb_sym_dict[p_r[0]]][1].append(p_r[2])
                    phrase_check = re.findall("CN\d+$",p_r[2])
                    if phrase_check:
                        verb_dict[verb_sym_dict[p_r[0]]][1].append(phrase_dict[p_r[2]])
                if verb_dict[verb_sym_dict[p_r[0]]][1] and verb_dict[verb_sym_dict[p_r[0]]][2]:
                    conj_words = verb_dict[verb_sym_dict[p_r[0]]][1]
                    for unit_ in verb_dict[verb_sym_dict[p_r[0]]][2]:
                        conjs_bef = ori_sentence.split(unit_)[0]
                        if all(word in conjs_bef for word in conj_words):
                            nearest = shortest_match(conj_words, conjs_bef)
                            # print(nearest)
                            if nearest:
                                verb_dict[verb_sym_dict[p_r[0]]][2].remove(unit_)
                                if nearest in verb_dict.keys():
                                    verb_dict[nearest][2].append(unit_)
                                else:
                                    verb_dict[nearest] = [[], [], []]
                                    verb_dict[nearest][2].append(unit_)

            if p_r[0] in verb_dict.keys():
                if "nsubj" in p_r[1]:
                    if p_r[2] in phrase_dict.keys():
                        if p_r[2] in rep_dict.keys():
                            verb_dict[p_r[0]][0].append(phrase_dict[rep_dict[p_r[2]]])
                        else:
                            verb_dict[p_r[0]][0].append(phrase_dict[p_r[2]])
                    elif p_r[2] in rep_dict.keys():
                        verb_dict[p_r[0]][0].append(rep_dict[p_r[2]])
                    else:
                        verb_dict[p_r[0]][0].append(p_r[2])
                if "obl" in p_r[1]:
                    unit_chunk = re.findall("UP\d+$", p_r[2])
                    if unit_chunk:
                        verb_dict[p_r[0]][2].append(unit_dict[p_r[2]])
                        # print(p_r[0],unit_dict[p_r[2]])
                        if unit_dict[p_r[2]] in copy_units:
                            copy_units.remove(unit_dict[p_r[2]])
                if "nmod" in p_r[1]:
                    unit_chunk = re.findall("UP\d+$", p_r[2])
                    if unit_chunk:
                        verb_dict[p_r[0]][2].append(unit_dict[p_r[2]])
                        # print(p_r[0],unit_dict[p_r[2]])
                        if unit_dict[p_r[2]] in copy_units:
                            copy_units.remove(unit_dict[p_r[2]])
                if "conj" in p_r[1]:
                    if p_r[2] in verbs:
                        verb_dict[p_r[0]][1].append(p_r[2])
                        # print(p_r[2], "222")
                    phrase_check = re.findall("CN\d+$",p_r[2])
                    if phrase_check:
                        verb_dict[p_r[0]][1].append(phrase_dict[p_r[2]])
                if verb_dict[p_r[0]][1] and verb_dict[p_r[0]][2]:
                    conj_words = verb_dict[p_r[0]][1]
                    for unit_ in verb_dict[p_r[0]][2]:

                        conjs_bef = ori_sentence.split(unit_)[0]
                        if all(word in conjs_bef for word in conj_words):
                            nearest = shortest_match(conj_words, conjs_bef)
                            # print(nearest)
                            if nearest:
                                verb_dict[p_r[0]][2].remove(unit_)
                                if nearest in verb_dict.keys():
                                    verb_dict[nearest][2].append(unit_)
                                else:
                                    verb_dict[nearest] = [[], [], []]
                                    verb_dict[nearest][2].append(unit_)
        for p_r in parse_results:
            phrase_chunk = re.findall("CN\d+$", p_r[0])
            if phrase_chunk:
                if "nsubj" in p_r[1]:
                    if p_r[2] in phrase_dict.keys():
                        if p_r[2] in rep_dict.keys():
                            verb_dict[phrase_dict[p_r[0]]][0].append(phrase_dict[rep_dict[p_r[2]]])
                        else:
                            verb_dict[phrase_dict[p_r[0]]][0].append(phrase_dict[p_r[2]])
                    elif p_r[2] in rep_dict.keys():
                        verb_dict[phrase_dict[p_r[0]]][0].append(rep_dict[p_r[2]])
                    else:
                        verb_dict[phrase_dict[p_r[0]]][0].append(p_r[2])
                if "nmod" in p_r[1]:
                    unit_chunk = re.findall("UP\d+$", p_r[2])
                    if unit_chunk:
                        verb_dict[phrase_dict[p_r[0]]][2].append(unit_dict[p_r[2]])
                        if unit_dict[p_r[2]] in copy_units:
                            copy_units.remove(unit_dict[p_r[2]])
                if "obl" in p_r[1]:
                    unit_chunk = re.findall("UP\d+$", p_r[2])
                    if unit_chunk:
                        # print(p_r)
                        verb_dict[phrase_dict[p_r[0]]][2].append(unit_dict[p_r[2]])
                        # print(p_r[0],unit_dict[p_r[2]])
                        if unit_dict[p_r[2]] in copy_units:
                            copy_units.remove(unit_dict[p_r[2]])
                if "compound" in p_r[1]:
                    unit_chunk = re.findall("UP\d+$", p_r[2])
                    if unit_chunk:
                        # print(p_r)
                        verb_dict[phrase_dict[p_r[0]]][2].append(unit_dict[p_r[2]])
                        # print(p_r[0],unit_dict[p_r[2]])
                        if unit_dict[p_r[2]] in copy_units:
                            copy_units.remove(unit_dict[p_r[2]])

                if "conj" in p_r[1]:
                    if p_r[2] in verbs:
                        verb_dict[phrase_dict[p_r[0]]][1].append(p_r[2])
                        # print(p_r[2], "333")
                    phrase_check = re.findall("CN\d+$", p_r[2])
                    if phrase_check:
                        verb_dict[phrase_dict[p_r[0]]][1].append(phrase_dict[p_r[2]])
                if p_r[0] in phrase_dict.keys():
                    if verb_dict[phrase_dict[p_r[0]]][1] and verb_dict[phrase_dict[p_r[0]]][2]:
                        conj_words = verb_dict[phrase_dict[p_r[0]]][1]
                        for unit_ in verb_dict[phrase_dict[p_r[0]]][2]:
                            conjs_bef = ori_sentence.split(unit_)[0]
                            if all(word in conjs_bef for word in conj_words):
                                nearest = shortest_match(conj_words, conjs_bef)
                                if nearest:
                                    verb_dict[phrase_dict[p_r[0]]][2].remove(unit_)
                                    if nearest in verb_dict.keys():
                                        verb_dict[nearest][2].append(unit_)
                                    else:
                                        verb_dict[nearest] = [[],[],[]]
                                        verb_dict[nearest][2].append(unit_)

        # all_method_action = list()
        # all_method_action.extend(verb_sym_dict.keys())
        # all_method_action.extend(sent_in_chunks)
        # # print(copy_units)
        # attribute_dict = dict()
        # all_attributes = list()
        # print(verb_dict)
        # print("copy_units:", copy_units)
        for p_r in parse_results:
            if "nmod" in p_r[1]:# nmod关系需要排在obl关系之前，因为参数一般都对应属性名是名词或者名词短语，很少会像heat那样动作直接跟参数
                unit_chunk = re.findall("UP\d+$", p_r[2])
                u_check = re.findall("UP\d+$", p_r[0])
                bef_part = sentence.split(p_r[2])[0]
                if p_r[0] in bef_part:
                    middle = bef_part.split(p_r[0])[-1]
                    middle = middle.strip()
                    mid_words = nltk.word_tokenize(middle)
                    # print(mid_words)
                # if all(a_m not in bef_part for a_m in act_labels_all):
                    # 只当两个词间的距离较近时我们相信依存解析树的解析结果,这里我们运用锚点对依存解析的结果进行跳转/远程关联
                    if len(mid_words)<=2:
                        if unit_chunk and p_r[0] not in verb_sym_dict.keys() and p_r[0] not in phrase_dict.keys() and not u_check and unit_dict[p_r[2]] in copy_units:
                            subj = None
                            reorg_results = list()
                            for p_r_o in parse_results:
                                reorg_results.append(p_r_o)
                                if p_r[2] == p_r_o[2]:
                                    break
                            reorg_results.reverse()
                            for p_r_o in reorg_results:
                                # if p_r_o[2] == p_r[0] and "nsubj" in p_r_o[1]:
                                #     if any(p_r[0] in sub for sub in phrase_dict.keys()) or any(p_r[0] in sub for sub in verb_sym_dict.keys()) :
                                #         # print(p_r_o,"000")
                                #         subj = True
                                #         if p_r_o[0] in phrase_dict.keys() and unit_dict[p_r[2]] not in verb_dict[phrase_dict[p_r_o[0]]][2]:
                                #             if unit_dict[p_r[2]] in copy_units:
                                #                 verb_dict[phrase_dict[p_r_o[0]]][2].append(unit_dict[p_r[2]])
                                #                 verb_dict[phrase_dict[p_r_o[0]]][0].append(p_r[0])
                                #                 copy_units.remove(unit_dict[p_r[2]])
                                #         elif p_r_o[0] in verb_sym_dict.keys() and unit_dict[p_r[2]] not in verb_dict[verb_sym_dict[p_r_o[0]]][2]:
                                #             verb_dict[verb_sym_dict[p_r_o[0]]][2].append(unit_dict[p_r[2]])
                                #             verb_dict[verb_sym_dict[p_r_o[0]]][0].append(p_r[0])
                                #             copy_units.remove(unit_dict[p_r[2]])
                                # obl和nmod的关系中，锚点词时位于第三位的
                                if p_r_o[2] == p_r[0] and "obl" in p_r_o[1]:#至于第二个这个排序是obl在nmod前面是因为工艺中的方法名称更习惯于直接在后面跟参数，而操作动词习惯接个属性名词再跟参数
                                    if any(p_r_o[0] in sub for sub in phrase_dict.keys()) or any(
                                            p_r_o[0] in sub for sub in verb_sym_dict.keys()):
                                        # print(p_r_o,"111")
                                        subj = True
                                        if p_r_o[0] in phrase_dict.keys() and unit_dict[p_r[2]] not in verb_dict[phrase_dict[p_r_o[0]]][2]:
                                            if unit_dict[p_r[2]] in copy_units:
                                                verb_dict[phrase_dict[p_r_o[0]]][2].append(unit_dict[p_r[2]])
                                                # verb_dict[phrase_dict[p_r_o[0]]][0].append(p_r[0])
                                                copy_units.remove(unit_dict[p_r[2]])
                                        elif p_r_o[0] in verb_sym_dict.keys() and unit_dict[p_r[2]] not in verb_dict[verb_sym_dict[p_r_o[0]]][2]:
                                            if unit_dict[p_r[2]] in copy_units:
                                                verb_dict[verb_sym_dict[p_r_o[0]]][2].append(unit_dict[p_r[2]])
                                                # verb_dict[verb_sym_dict[p_r_o[0]]][0].append(p_r[0])
                                                copy_units.remove(unit_dict[p_r[2]])
                                        # print(verb_dict)
                                elif p_r_o[2] == p_r[0] and "nmod" in p_r_o[1]:
                                    if any(p_r_o[0] in sub for sub in phrase_dict.keys()) or any(
                                            p_r_o[0] in sub for sub in verb_sym_dict.keys()):
                                        # print(p_r_o,"222")
                                        subj = True
                                        if p_r_o[0] in phrase_dict.keys() and unit_dict[p_r[2]] not in verb_dict[phrase_dict[p_r_o[0]]][2]:
                                            if unit_dict[p_r[2]] in copy_units:
                                                verb_dict[phrase_dict[p_r_o[0]]][2].append(unit_dict[p_r[2]])
                                                # verb_dict[phrase_dict[p_r_o[0]]][0].append(p_r[0])
                                                copy_units.remove(unit_dict[p_r[2]])
                                        elif p_r_o[0] in verb_sym_dict.keys() and unit_dict[p_r[2]] not in verb_dict[verb_sym_dict[p_r_o[0]]][2]:
                                            if unit_dict[p_r[2]] in copy_units:
                                                verb_dict[verb_sym_dict[p_r_o[0]]][2].append(unit_dict[p_r[2]])
                                                # verb_dict[verb_sym_dict[p_r_o[0]]][0].append(p_r[0])
                                                copy_units.remove(unit_dict[p_r[2]])

                            if not subj and p_r[0] not in unit_dict.keys():
                                # print(p_r[0],"333")
                                all_verb_chunk.append(p_r[0])
                                # all_attributes.append(p_r[0])
                                verb_dict[p_r[0]] = [[],[],[]]
                                verb_dict[p_r[0]][2].append(unit_dict[p_r[2]])
                                if unit_dict[p_r[2]] in copy_units:
                                    copy_units.remove(unit_dict[p_r[2]])

                        # attribute_dict[p_r[0]] = [[], [], []]
                        # attribute_dict[p_r[0]][2].append(unit_dict[p_r[2]])
                        # if unit_dict[p_r[2]] in copy_units:
                        #     copy_units.remove(unit_dict[p_r[2]])

                        elif unit_chunk and p_r[0] in phrase_dict.keys() and unit_dict[p_r[2]] in copy_units:
                            verb_dict[phrase_dict[p_r[0]]][2].append(unit_dict[p_r[2]])
                            if unit_dict[p_r[2]] in copy_units:
                                copy_units.remove(unit_dict[p_r[2]])
            if "obl" in p_r[1]:
                unit_chunk = re.findall("UP\d+$", p_r[2])
                bef_part = sentence.split(p_r[2])[0]
                if p_r[0] in bef_part:
                    middle = bef_part.split(p_r[0])[-1]
                    middle = middle.strip()
                    mid_words = nltk.word_tokenize(middle)
                    # if all(a_m not in bef_part for a_m in act_labels_all):
                    if len(mid_words) <= 2:
                        # if all(a_m not in bef_part for a_m in act_labels_all):
                        if unit_chunk and p_r[0] not in verb_sym_dict.keys() and p_r[0] not in phrase_dict.keys() and unit_dict[p_r[2]] in copy_units:
                            subj = None
                            reorg_results = list()
                            for p_r_o in parse_results:
                                reorg_results.append(p_r_o)
                                if p_r[2] == p_r_o[2]:
                                    break
                            reorg_results.reverse()
                            for p_r_o in reorg_results:
                                if p_r_o[0] == p_r[0] and "nsubj" in p_r_o[1]:
                                    # print(p_r_o)
                                    if any(p_r_o[2] in sub for sub in phrase_dict.keys()) or any(p_r_o[2] in sub for sub in verb_sym_dict.keys()) :
                                        subj = True
                                        if p_r_o[2] in phrase_dict.keys():
                                            if unit_dict[p_r[2]] not in verb_dict[phrase_dict[p_r_o[2]]][2]:
                                                if unit_dict[p_r[2]] in copy_units:
                                                    # print(phrase_dict[p_r_o[2]], unit_dict[p_r[2]])
                                                    verb_dict[phrase_dict[p_r_o[2]]][2].append(unit_dict[p_r[2]])
                                                    # verb_dict[phrase_dict[p_r_o[2]]][0].append(p_r[0])
                                                    copy_units.remove(unit_dict[p_r[2]])
                                                    break
                                        elif p_r_o[2] in verb_sym_dict.keys():
                                            if unit_dict[p_r[2]] not in verb_dict[verb_sym_dict[p_r_o[2]]][2]:
                                                if unit_dict[p_r[2]] in copy_units:
                                                    verb_dict[verb_sym_dict[p_r_o[2]]][2].append(unit_dict[p_r[2]])
                                                    # verb_dict[verb_sym_dict[p_r_o[0]]][0].append(p_r[0])
                                                    copy_units.remove(unit_dict[p_r[2]])
                                                    break
                                elif p_r_o[2] == p_r[0] and "obl" in p_r_o[1]:#至于第二个这个排序是obl在nmod前面是因为工艺中的方法名称更习惯于直接在后面跟参数，而操作动词习惯接个属性名词再跟参数
                                    if any(p_r_o[0] in sub for sub in phrase_dict.keys()) or any(
                                            p_r_o[0] in sub for sub in verb_sym_dict.keys()):
                                        # print(p_r_o,"111")
                                        subj = True
                                        if p_r_o[0] in phrase_dict.keys() and unit_dict[p_r[2]] not in verb_dict[phrase_dict[p_r_o[0]]][2]:
                                            if unit_dict[p_r[2]] in copy_units:
                                                verb_dict[phrase_dict[p_r_o[0]]][2].append(unit_dict[p_r[2]])
                                                # verb_dict[phrase_dict[p_r_o[0]]][0].append(p_r[0])
                                                copy_units.remove(unit_dict[p_r[2]])
                                        elif p_r_o[0] in verb_sym_dict.keys() and unit_dict[p_r[2]] not in verb_dict[verb_sym_dict[p_r_o[0]]][2]:
                                            if unit_dict[p_r[2]] in copy_units:
                                                verb_dict[verb_sym_dict[p_r_o[0]]][2].append(unit_dict[p_r[2]])
                                                # verb_dict[verb_sym_dict[p_r_o[0]]][0].append(p_r[0])
                                                copy_units.remove(unit_dict[p_r[2]])
                                        # print(verb_dict)
                                elif p_r_o[2] == p_r[0] and "nmod" in p_r_o[1]:
                                    if any(p_r_o[0] in sub for sub in phrase_dict.keys()) or any(
                                            p_r_o[0] in sub for sub in verb_sym_dict.keys()):
                                        # print(p_r_o,"222")
                                        subj = True
                                        if p_r_o[0] in phrase_dict.keys() and unit_dict[p_r[2]] not in verb_dict[phrase_dict[p_r_o[0]]][2]:
                                            if unit_dict[p_r[2]] in copy_units:
                                                verb_dict[phrase_dict[p_r_o[0]]][2].append(unit_dict[p_r[2]])
                                                # verb_dict[phrase_dict[p_r_o[0]]][0].append(p_r[0])
                                                copy_units.remove(unit_dict[p_r[2]])
                                        elif p_r_o[0] in verb_sym_dict.keys() and unit_dict[p_r[2]] not in verb_dict[verb_sym_dict[p_r_o[0]]][2]:
                                            if unit_dict[p_r[2]] in copy_units:
                                                verb_dict[verb_sym_dict[p_r_o[0]]][2].append(unit_dict[p_r[2]])
                                                # verb_dict[verb_sym_dict[p_r_o[0]]][0].append(p_r[0])
                                                copy_units.remove(unit_dict[p_r[2]])

                            if not subj and p_r[0] not in unit_dict.keys():
                                all_verb_chunk.append(p_r[0])
                                verb_dict[p_r[0]] = [[], [], []]
                                verb_dict[p_r[0]][2].append(unit_dict[p_r[2]])
                                if unit_dict[p_r[2]] in copy_units:
                                    copy_units.remove(unit_dict[p_r[2]])
                                if unit_dict[p_r[2]] in copy_units:
                                    copy_units.remove(unit_dict[p_r[2]])
                        elif unit_chunk and p_r[0] in phrase_dict.keys() and unit_dict[p_r[2]] in copy_units:
                            verb_dict[phrase_dict[p_r[0]]][2].append(unit_dict[p_r[2]])
                            if unit_dict[p_r[2]] in copy_units:
                                copy_units.remove(unit_dict[p_r[2]])


        # print(ori_sentence)
        # print(all_verb_chunk)
        # print(verb_dict.keys())
        # print(copy_units)
        # print(sentence)
        # print(verb_dict)
        # print(ori_sentence)
        # print("copy_units:",copy_units)
        if all_verb_chunk and copy_units:
            for label,unit in unit_dict.items():
                sentence = sentence.replace(" "+label+" "," "+unit+" ")
                ori_sentence = ori_sentence.replace(unit," "+unit+" ")
            # print(sentence)
            mathch_unit =list()
            for other_unit in copy_units:
                if " "+other_unit+" " in sentence:
                    num_unit = mathch_unit.count(other_unit)
                    bef_part = ori_sentence.split(" "+other_unit+" ")[:num_unit+1]
                    # print(other_unit,bef_part)
                    if isinstance(bef_part,list):
                        bef_part = " ".join(bef_part)
                    nearest = shortest_match(verb_dict.keys(), bef_part)
                    # print(nearest)

                    mathch_unit.append(other_unit)
                    if nearest and nearest in verb_dict.keys():
                        verb_dict[nearest][2].append(other_unit)

            # 是否将只要被识别出的任意一个属性或者动词的主语赋予其它操作
            # pre_sample = None
            # for verb,info in verb_dict.items():
            #     if info[0]:
            #         pre_sample = info[0][0]
            #     if not info[0] and pre_sample:
            #         verb_dict[verb][0].append(pre_sample)

        out_verb_dict = copy.deepcopy(verb_dict)
        # print(out_verb_dict,"222")
        for verb,info in verb_dict.items():
            if verb in other_chunks and not verb_dict[verb][2] and verb not in chunks_dict and verb not in tokens_dict:
                out_verb_dict.pop(verb)
        order_verb_dict = dict()
        verb_order = list()
        verb_space = list()
        for verb,info in out_verb_dict.items():
            verb_order.append(verb)
            verb_space.append(len(ori_sentence.split(verb)[0]))
        # print(out_verb_dict)
        for sp in sorted(verb_space):
            order_verb_dict[verb_order[verb_space.index(sp)]] = out_verb_dict[verb_order[verb_space.index(sp)]]
    return order_verb_dict

def dp_result(nlp,sentence,tokens_dict,chunk_dict, wnl_chunk_dicts,wnl_token_dicts,replace_items,unit_list,symbols):
    all_verbs = list()
    sentence = re.sub(r"\\u.{4}","",sentence)
    sent_in_tokens,sent_in_chunks,extra_chunks = get_candidated_action(sentence, tokens_dict, chunk_dict, wnl_chunk_dicts, wnl_token_dicts, nlp,unit_list)
    sent_unit,sentence,units_relas = sent_unit_parsing(sentence,replace_items,unit_list,symbols)
    # print(sent_in_tokens,sent_in_chunks)

    outcome_dict = dp_parsing(tokens_dict,chunk_dict,nlp, sentence,sent_in_tokens,sent_in_chunks,sent_unit,extra_chunks,symbols)
    # print(outcome_dict)
    if outcome_dict:
        mat_bef = None
        # print(outcome_dict,444)
        for verb,info in outcome_dict.items():
            output_v = dict()
            if verb in sent_in_tokens or verb in sent_in_chunks:
                output_v["action"] = verb
                if info[0]:
                    output_v["mat"] = info[0]
                    mat_bef = info[0]
                elif not info[0] and mat_bef:
                    output_v["mat"] = mat_bef
                else:
                    output_v["mat"] = list()
                output_v["units"] = info[2]
            else:
                if info[2]:
                    output_v["attribute"] = verb
                    if info[0]:
                        output_v["mat"] = info[0]
                    else:
                        output_v["mat"] = list()
                    output_v["units"] = info[2]
                else: #若动作不存在于token或者chunk词典中以及该动作没有任何依存的参数时，舍去
                    continue
            if units_relas:
                new_unit_rela = list()
                for u_r in units_relas:
                    if "units" in output_v.keys():
                        if any(unit in u_r for unit in output_v["units"]):
                            new_unit_rela.append(u_r)
                output_v["unit_rela"] = new_unit_rela
            else:
                output_v["unit_rela"] = None

            all_verbs.append(output_v)
    return all_verbs

# 从excel表格中得到所有的doi和段落信息
def sentences_read(excel_path):
    # 将信息从excel中拿出来放入json文件中,每个key下包含了所有相关的段落信息
    lower_chunks = list()
    xls_sp = xlrd.open_workbook(excel_path)
    sht_sp = xls_sp.sheet_by_index(0)
    doi_sp = sht_sp.col_values(1)
    parag_sp = sht_sp.col_values(0)
    all_info = dict()
    for info_i in range(len(doi_sp)):
        if doi_sp[info_i] not in all_info.keys():
            all_info[doi_sp[info_i]] = list()
            all_info[doi_sp[info_i]].append(parag_sp[info_i])
        else:
            all_info[doi_sp[info_i]].append(parag_sp[info_i])
    return all_info

def parag_mat(sents,mats,verb_info):
    if "action" in verb_info.keys():
        verb = verb_info["action"]
    elif "attribute" in verb_info.keys():
        verb = verb_info["attribute"]
    output = None
    parags = " ".join(sents)
    bef = parags.split(verb)[0]
    mats_len = list()
    in_mats = list()
    for mat in mats:
        if isinstance(mat, str):
            if " "+mat+" " in bef:
                # print(" "+mat+" ", bef,"here!!!")
                len_m = len(bef.split(" "+mat+" ")[-1])
                mats_len.append(len_m)
                in_mats.append(mat)
    if mats_len and in_mats:
        mat_ind = mats_len.index(min(mats_len))
        output = in_mats[mat_ind]
    return output

def sent_replace(sent_names,part):
    for name in sent_names:
        sub_w = nltk.word_tokenize(name)
        if name in part and len(sub_w) > 1:
            label = name.replace(" ","_")
            part = part.replace(name,label)
            sent_names.remove(name)
            sent_names.append(label)
    return sent_names,part

def get_nearest_name(sent_names,sent_parts,unit):
    # print(sent_names,sent_parts,unit)
    nearest_name = None
    for part in sent_parts:
        unit_id = None
        if unit in part and any(name in part for name in sent_names):
            sent_names,part = sent_replace(sent_names,part)
            # print(sent_names,part)
            part_words = part.split(" ")
            names = list()
            names_id = list()
            for id_w,word in enumerate(part_words):
                if word == unit or word.startswith(unit):
                    unit_id = id_w
                elif word in sent_names or any(word.startswith(name) for name in sent_names):
                    names.append(word)
                    names_id.append(id_w)
            # print(names)
            # print(names_id)
            if unit_id and names_id:
                id_dv = list()
                for n_i in range(len(names_id)):
                    id_dv.append(abs(names_id[n_i]-unit_id))
                # print(id_dv,"1111")
                # print(sent_names)
                # print(part)
                # print(names_id)
                nearest_name = names[id_dv.index(min(id_dv))]
                if nearest_name:
                    for word in sent_names:
                        if word.startswith(nearest_name):
                            sent_names.remove(word)
                            break
    return nearest_name, sent_names

def re_order(sent,all_verbs,verb_i,sent_names):
    units = all_verbs[verb_i]["units"]
    sent_parts = sent.split(",")
    mat_units = dict()
    for unit in units:
        nearest_name,sent_names = get_nearest_name(sent_names,sent_parts,unit)
        # print(unit,nearest_name)
        if nearest_name:
            if nearest_name not in mat_units.keys():
                nearest_name = nearest_name.replace("_"," ")
                mat_units[nearest_name] = list()
                mat_units[nearest_name].append(unit)
            else:
                mat_units[nearest_name].append(unit)
    all_verbs[verb_i]["mat_units"] = mat_units
    # print(all_verbs[verb_i])
    return all_verbs

def comp_text_extractor(sent,ele_list,units):
    words = nltk.word_tokenize(sent)
    sent_info = dict()
    output = list()
    sent = sent.replace("("," ( ")
    sent = sent.replace(")", ",) ")
    sent = sent.replace(" "," ")
    sent = sent.replace(" ", " ")
    sent = sent.replace("nickel-", "Ni-")
    sent = sent.replace("nickel", "Ni")
    sent = sent.replace("cobalt-", "Co-")
    sent = sent.replace("cobalt", "Co")
    sent = re.sub("\s+"," ",sent)
    ele_unique = None
    one_ele_unique = None
    if "composition" in sent or "component" in sent:
        if sent.endswith("bal.") or sent.endswith("Bal."):
            sent += "."
        outcome_1 = re.findall("[><]*[\d\.]*[A-Z][a-z]?[\–\-][\d\.]+[A-Z][a-z]?[\–\-][^,\f\n\r\t\v\s]+", sent)
        # print(outcome_1,"111")
        for num_0 in outcome_1:
            x_comp = None
            for ele in ele_list:
                if num_0 + " " +"x"+" "+ele in sent:
                    output.append(num_0 + "-" +"x"+ele)
                    x_comp = True
                elif num_0 + " " +"x"+ele in sent:
                    output.append(num_0 + "-" +"x"+ele)
                    x_comp = True
                elif num_0 + " " +ele in sent:
                    output.append(num_0 +" "+ele)
                    x_comp = True
            if not x_comp:
                ele_in_num = 0
                for ele in ele_list:
                    if ele in num_0:
                        ele_in_num += 1
                if ele_in_num > 2:
                    output.append(num_0)

        check = re.findall("[^\da-zA-Z][><]*[\d]{1,2}[\.\d\~]{0,6}[aw]?t?\.?%?[\sto\s\-\–\–\~]*[\d]{0,2}[\.\d]{0,6}\s?[aw]?t?\.?%?|[Bb]ala?n?c?e?\.?|x", sent)
        for del_i in range(len(check)):
            if check.count("x") > 1:
                check.remove("x")
            else:
                break
        outcome_2 = list()
        to_ele_list = copy.deepcopy(ele_list)
        for num in check:
            if not num.endswith("%"):
                if num.endswith("a") or num.endswith("w"):
                    num = re.sub("at?|wt?", "", num)
            check_num = None
            num = re.sub("[^\da-zA-Z\.><%\-\–\~]", "", num)#其中涉及的字符是成分写法中可能包括的
            if "." in num:
                check_num = re.findall("\d{3}\.",num)
            else:
                check_num = re.findall("\d{3}", num)

            if not check_num:
                for ele in to_ele_list:#" " +
                    if any(num + unit + ele+',' in sent for unit in units) or any(num + unit + ele+';' in sent for unit in units) or any(num + unit + ele+'.' in sent for unit in units) or any(num + unit + ele+' and' in sent for unit in units) or any('and ' +num + unit + ele in sent for unit in units) or any(', '+num + unit + ele in sent for unit in units) or any('; '+num + unit + ele in sent for unit in units):
                        check_in = None
                        for unit in units:
                            if num + unit + ele in sent:
                                bef_part_end = sent.split(num + unit + ele)[0][-1]
                                bef_check = re.findall("[\d\.]", bef_part_end)
                                if not bef_check:
                                    check_in = True
                                    break
                        if check_in:
                            outcome_2.append(num + " " + ele)
                            to_ele_list.remove(ele)
                            break
                    elif num + ele+',' in sent or num + ele+';' in sent or num + ele+'.' in sent or num + ele+' and' in sent or 'and ' + num + ele in sent or ', ' + num + ele in sent or '; '+num + ele in sent:
                        check_in = None
                        bef_part_end = sent.split(num + ele)[0][-1]
                        bef_check = re.findall("[\d\.]", bef_part_end)
                        if not bef_check:
                            check_in = True
                        if check_in:
                            outcome_2.append(num + " " + ele)
                            to_ele_list.remove(ele)
                            break
                    elif any(ele + unit + num+',' in sent for unit in units) or any(ele + unit + num+';' in sent for unit in units) or any(ele + unit + num+'.' in sent for unit in units) or any(ele + unit + num+' and' in sent for unit in units) or any('and ' + ele + unit + num in sent for unit in units) or any(', '+ele + unit + num in sent for unit in units) or any('; '+ele + unit + num in sent for unit in units):
                        outcome_2.append(ele + " " + num)
                        to_ele_list.remove(ele)
                        break
                    elif ele + num+',' in sent or ele + num+';' in sent or ele + num+'.' in sent or ele + num+' and' in sent or 'and '+ele + num in sent or ', '+ele + num in sent or '; '+ele + num in sent:
                        outcome_2.append(ele + " " + num)
                        to_ele_list.remove(ele)
                        break
                bracket_part = re.findall("\([a-zA-Z\s\+\,]+\)",sent)
                brack_eles = list()
                if bracket_part:
                    for b_p in bracket_part:
                        if "+" in b_p:
                            in_ele = 0
                            for ele in to_ele_list:
                                if ele in b_p:
                                    in_ele += 1
                                    brack_eles.append(ele)
                            if in_ele >1:
                                if num + b_p in sent or num + " " + b_p in sent:

                                    b_p = b_p.replace(",","")
                                    brack_ele_num = num + " " + b_p
                                    outcome_2.append(brack_ele_num)
                                    for brack_e in brack_eles:
                                        to_ele_list.remove(brack_e)

        outcome_2 = list(set(outcome_2))
        if outcome_1:
            one_ele_unique = True
        if len(outcome_2) > 2:
            output.extend(outcome_2)
            ele_unique = True
        if "wt%" in sent or "wt.%" in sent or "weight percentage" in sent or "weight percent" in sent or "mass" in sent:
            sent_info["unit"] = "wt.%"
        elif "at%" in sent or "at.%" in sent or "atomic percentage" in sent or "atomic percent" in sent:
            sent_info["unit"] = "at.%"
        if output:
            sent_info["composition"] = output
        main_ele = list()
        for ele in ele_list:
            if ele in words and all(ele not in ele_num for ele_num in output):
                main_ele.append(ele)
        if ele_unique:
            if all("bal" not in ele_num for ele_num in output) and all("Bal" not in ele_num for ele_num in output):
                if main_ele:
                    sent_info["main_ele"] = main_ele
                else:
                    for word in words:
                        if "-base" in word:
                            main_ele.append(word)
                            sent_info["main_ele"] = main_ele
                            break
        if one_ele_unique:
            if "bal" not in output and "Bal" not in output:
                sent_info["main_ele"] = main_ele
    return sent_info

def parag_info(nlp, parags, tokens_dict, chunks_dict,replace_items,unit_list,symbols,ele_list,units):
    wnl_chunk_dicts = wnl_chunks(chunks_dict)
    wnl_token_dicts = wnl_tokens(tokens_dict)
    composition_from_text = list()

    para = Paragraph(str(parags))
    for par in para:
        sent = par.text
        sent_info = comp_text_extractor(sent,ele_list,units)
        if sent_info:
            composition_from_text.append(str(sent_info))
        composition_from_text = list(set(composition_from_text))
    sents = list()
    for par in para:
        sents.append(par.text)
    sent_info = dict()
    for sent in sents:
        all_verbs = dp_result(nlp, sent, tokens_dict, chunks_dict, wnl_chunk_dicts, wnl_token_dicts,replace_items,unit_list,symbols)
        for verb_i in range(len(all_verbs)):
            if "composition" not in all_verbs[verb_i].keys() and composition_from_text:
                all_verbs[verb_i]["composition"] = composition_from_text
        if all_verbs:
            sent_info[sent] = all_verbs
    return sent_info

class ner_dp:
    def __init__(self, parags, java_path,stanford_parser_path,stanford_model_path,token_dict_path,chunk_dict_path,nlp_path,replace_items,unit_list,
                 symbols,ele_list,units):
        self.parags = parags
        self.java_path = java_path
        self.stanford_parser_path = stanford_parser_path
        self.stanford_model_path = stanford_model_path
        self.token_dict_path = token_dict_path
        self.chunk_dict_path = chunk_dict_path
        self.nlp_path = nlp_path
        self.replace_items = replace_items
        self.unit_list = unit_list
        self.symbols = symbols
        self.ele_list = ele_list
        self.units = units

    def run(self):
        with open(self.token_dict_path, 'r', encoding='utf-8') as ft:
            tokens_dict = json.load(ft)
        new_tokens = list()
        for token in tokens_dict:
            new_tokens.append(token.lower())
        token_dict = new_tokens
        with open(self.chunk_dict_path, encoding="utf-8", mode="r") as file:
            chunks_dict = json.load(file)
        new_chunks = list()
        for chunk in chunks_dict:
            new_chunks.append(chunk.lower())
        chunks_dict = new_chunks
        nlp = StanfordCoreNLP(self.nlp_path)
        parsing_results = parag_info(nlp, self.parags, token_dict, chunks_dict,self.replace_items,self.unit_list,self.symbols,self.ele_list,self.units)
        nlp.close()
        return parsing_results

def main(parags,config_path,java_path,stanford_parser_path,stanford_model_path,nlp_path,token_dict_path,chunk_dict_path,output_path):
    infile = cp.ConfigParser()
    infile.read(config_path,"UTF-8")
    unit_list = eval(infile.get("SYNTHESIS_EXTRACTOR","unit_list"))
    symbols = eval(infile.get("SYNTHESIS_EXTRACTOR","symbols"))
    replace_items = eval(infile.get("SYNTHESIS_EXTRACTOR","replace_items"))
    ele_list = eval(infile.get("SYNTHESIS_EXTRACTOR","ele_list"))
    units = eval(infile.get("SYNTHESIS_EXTRACTOR","units"))
    parse = ner_dp(parags, java_path,stanford_parser_path,stanford_model_path,token_dict_path,chunk_dict_path,nlp_path,replace_items,unit_list,symbols,
                   ele_list, units)
    parsing_results = parse.run()
    with open(output_path, "w+", encoding="utf-8") as f:
        f.write(json.dumps(parsing_results))
    return parsing_results

config_path = r"C:\Users\Administrator\Desktop\wwr-files\web_code\test_files\dictionary.ini"
java_path = r"C:\Program Files\Java\jre1.8.0_181\bin/java.exe"
stanford_parser_path = r"C:\Users\Administrator\Desktop\wwr-files\code\files\stanford-parser.jar"
stanford_model_path = r"C:\Users\Administrator\Desktop\wwr-files\code\files/stanford-parser-4.2.0-models.jar"
nlp_path = r"C:\Users\Administrator\Desktop\wwr-files\code\stanford-corenlp-4.4.0"
token_dict_path = r'C:\Users\Administrator\Desktop\wwr-files\code\files\dict\tokens.json'
chunk_dict_path = r"C:\Users\Administrator\Desktop\wwr-files\code\chunk_output\save-new\heads\recom\params_optimization\refresh_noheads\refresh\modify_high_1.json"

parags = "Samples were exposed at 1300°F and 1400°F (704°C and 760°C) in both ambient laboratory air and in air containing a deliberate addition of water vapor."

output_path = r"C:\Users\Administrator\Desktop\wwr-files\code\output\refresh_table_strategy\results_modified\results.json"
parsing_results = main(parags,config_path,java_path,stanford_parser_path,stanford_model_path,nlp_path,token_dict_path,chunk_dict_path,output_path)

