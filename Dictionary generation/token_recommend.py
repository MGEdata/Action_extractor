# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 13:39:20 2022

@author: wwr
"""

import os
from gensim.models import word2vec
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import tqdm
import re
from tqdm import tqdm
from .normalization import stopword_list
import nltk
import fasttext
import random

class Load_model:
    def __init__(self, path, prop_name, n):
        self.path = path
        self.prop_name = prop_name
        self.n = n

    def load(self):
        model = word2vec.Word2Vec.load(self.path)
        new_sim_list = list()
        if len((self.prop_name).split(" ")) == 1:
            similarity_list = model.wv.most_similar(self.prop_name, topn=self.n)
            for s_word in similarity_list:
                check = re.findall("^[a-zA-Z\-\/]$", s_word)
                if check:
                    new_sim_list.append(s_word)
        else:
            # If the input is a word block, then the last word is selected as the target to generate the word vector
            similarity_list = model.wv.most_similar((self.prop_name).split(" ")[-1], topn=self.n)
            for s_word in similarity_list:
                check = re.findall("^[a-zA-Z\-\/]$", s_word)
                if check:
                    new_sim_list.append(s_word)
        return new_sim_list

def ave_vec_sims(word,words,model_path):
    model = word2vec.Word2Vec.load(model_path)
    word_wv = model.wv.get_vector(word).reshape(1,-1)
    ave_sims = 0
    for s_w in words:
        s_word_wv = model.wv.get_vector(s_w).reshape(1,-1)
        sim = cosine_similarity(word_wv, s_word_wv)
        ave_sims += sim[0][0]
    ave_sims = ave_sims/len(words)
    return ave_sims

def nn_lem_word(seeds, fasttext_recomm, verb_lemma):
    seeds_lem = list()
    output_words = list()
    for seed in seeds:
        lem = verb_lemma[seed][0]
        vb = verb_lemma[seed][1]
        if lem:
            seeds_lem.append(lem)

        elif vb:
            seeds_lem.append(vb)
    for fas_sub in fasttext_recomm:
        lem = verb_lemma[fas_sub][0]
        vb = verb_lemma[fas_sub][1]
        if lem:
            if lem in seeds_lem or any(lem.endswith(seed_l) for seed_l in seeds_lem):
                output_words.append(fas_sub)
        elif vb:
            if vb in seeds_lem or any(vb.endswith(seed_l) for seed_l in seeds_lem):
                output_words.append(fas_sub)

    return output_words

def countX(lst, x):
    return lst.count(x)

def was_were_filter(generated_words, sentences):
    """
    Secondary screening of process verbs initially recommended by the word vector based on the presence or absence of “was/were + V-ed” in all corpus
    """
    new_generated_words = list()
    for word in generated_words:
        if not word.endswith("ing"):
            if "was " + word + " " in sentences or "was " + word + "." in sentences or "was " + word + "," in sentences or "were " + word + " " in sentences or "were " + word + "." in sentences or "were " + word + "," in sentences:
                new_generated_words.append(word)
        else:
            new_generated_words.append(word)
    return new_generated_words

def reject_filter(generated_words, rejected_words):
    new_gen = list()
    for r_w in generated_words:
        if r_w not in rejected_words:
            new_gen.append(r_w)
    return new_gen

def fasttext_sims(model_path, t_word, all_verbs, sim_thre, seeds):
    model = fasttext.load_model(model_path)
    tar = model.get_word_vector(t_word).reshape(1, -1)
    to_words = list()

    for verb in all_verbs:
        vec = model.get_word_vector(verb).reshape(1, -1)
        sim = cosine_similarity(tar, vec)
        if sim[0][0] > sim_thre and verb not in seeds:
            to_words.append(verb)

    return to_words

def word2vec_sim_filter(model, word, sim):
    recom_words = list()
    seed_sims = model.wv.most_similar(word, topn=300)
    for word in seed_sims:
        if word[1] >= sim:
            recom_words.append(word[0])
        else:
            break
    return recom_words

def get_top_20(sim_dict):
    lis = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
    words = list()
    for item in lis:
        words.append(item[0])
    return words

def fasttext_not_vb_check(token, sent, sentences_tag):
    tagged_sentence = sentences_tag[sent]
    tags = list()
    words = list()
    recommed_check = None
    for sub_p in tagged_sentence:
        if sub_p[1] != "DT" and sub_p[1] != "CD":
            words.append(sub_p[0])
            tags.append(sub_p[1])
    for w_i in range(len(words)):
        if words[w_i] == token:
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
                        b_check = re.findall("VB[DNP]\sTO$", b_tags)
                        a_check = re.findall("^NN|^JJ|VB[DN]?$|CD", aft_tags)
                        if b_check and not a_check and any(
                                w_b in bef_words for w_b in
                                bef_existed):
                            recommed_check = True
            if tags[w_i] == "VBD" or tags[w_i] == "VBN":
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
                        b_check = re.findall("VB[DNP][\sRB]*$", b_tags)
                        b_and_check = re.findall("VB[DNP]\sCC[\sRB]*$|CC[\sRB]*$", b_tags)
                        b_in_check = re.findall("CC[\sRB]*$", b_tags)
                        a_check = re.findall("^NN|^JJ|VB[DN]?$|CD", aft_tags)
                        a_sym_check = re.findall("[a-zA-Z]+", aft_tags)
                        if b_check and not a_check and any(
                                w_b in bef_words for w_b in
                                bef_existed):
                            recommed_check = True
                        if b_and_check and not a_check:
                            recommed_check = True
                        if b_in_check and aft_tags.startswith("IN"):
                            recommed_check = True
                        if b_in_check and not a_sym_check:
                            recommed_check = True
                    elif len(bef_words) == 2:
                        b_check = re.findall("NN.*\sVB[DNP]$", b_tags)
                        a_check = re.findall("NN|JJ|VB[DN]?$|CD", aft_tags)
                        if b_check and not a_check and any(w_b in bef_words for w_b in bef_existed):
                            recommed_check = True
            elif tags[w_i] == "JJ":
                if token.endswith("ed") or token.endswith("ing"):
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
                            b_check = re.findall("VB[DNP][\sRB]*$", b_tags)
                            b_and_check = re.findall("VB[DNP]\sCC[\sRB]*$", b_tags)
                            a_check = re.findall("^NN|^JJ|VB[DN]?$|CD", aft_tags)
                            if b_check and not a_check and any(
                                    w_b in bef_words for w_b in
                                    bef_existed):
                                recommed_check = True
                            if b_and_check and not a_check:
                                recommed_check = True
                        elif len(bef_words) == 2:
                            b_check = re.findall("NN.\sVB[DNP]$|PRP\sVB[DNP]$", b_tags)
                            a_check = re.findall("NN|JJ|VB[DN]?$|CD", aft_tags)
                            if b_check and not a_check and any(w_b in bef_words for w_b in bef_existed):
                                recommed_check = True

            elif tags[w_i] == "VBG":
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
                    if tags_b_words:
                        b_check_2 = re.findall("^VB[DNP]\sIN$", tags_b_words)
                        b_check_3 = re.findall("IN$|CC$", tags_b_words)
                    elif bef_tags_1:
                        b_check_1 = re.findall("IN", bef_tags_1)
                    if aft_tags:
                        a_check = re.findall("NN|JJ|VB[DN]?$|CD", aft_tags)
                        a_check_3 = re.findall("IN", aft_tags)
                    if len(bef_words) == 1:
                        if not a_check and b_check_1:
                            recommed_check = True
                    if len(bef_words) == 2:
                        if not a_check and b_check_2:
                            recommed_check = True
                        if b_check_3 and a_check_3:
                            recommed_check = True
            elif nn_check and token.endswith("ing"):
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
                if bef_words and aft_words:
                    if tags_b_words:
                        b_check_2 = re.findall("^VB[DN]\sIN$|CC$", tags_b_words)
                    elif bef_tags_1:
                        b_check_1 = re.findall("IN", bef_tags_1)
                    if aft_tags:
                        a_check = re.findall("NN|JJ|VB[DN]$|CD", aft_tags)
                    if len(bef_words) == 1:
                        if not a_check and b_check_1:
                            recommed_check = True
                    if len(bef_words) == 2:
                        if not a_check and b_check_2:
                            recommed_check = True

    return recommed_check

def chunk_window(token, sent, bef_existed, sentences_tag):
    tagged_sentence = sentences_tag[sent]
    tags = list()
    words = list()
    recommed_check = None
    nn_check = None
    for sub_p in tagged_sentence:
        if sub_p[1] != "DT" and sub_p[1] != "CD":
            words.append(sub_p[0])
            tags.append(sub_p[1])
    for w_i in range(len(words)):
        if words[w_i] == token:
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
                        b_check = re.findall("VB[DNP]\sTO$|CC\s[RB]*$", b_tags)
                        a_check = re.findall("^NN|^JJ|VB[DN]?$|CD", aft_tags)
                        if b_check and not a_check and any(
                                w_b in bef_words for w_b in
                                bef_existed):
                            recommed_check = True
            if tags[w_i] == "VBD" or tags[w_i] == "VBN":
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
                        b_check = re.findall("VB[DNP][\sRB]*$", b_tags)
                        b_and_check = re.findall("VB[DNP]\sCC\s[RB]*$|CC\s[RB]*$", b_tags)
                        a_check = re.findall("NN|JJ|VB[DN]?$|CD", aft_tags)
                        a_sym_check = re.findall("[a-zA-Z]+", aft_tags)
                        b_in_check = re.findall("CC[\sRB]*$", b_tags)
                        if b_check and not a_check and any(w_b in bef_words for w_b in
                                                           bef_existed):
                            recommed_check = True
                        if b_and_check and not a_check:
                            recommed_check = True
                        if b_in_check and aft_tags.startswith("IN"):
                            recommed_check = True
                        if b_in_check and not a_sym_check:
                            recommed_check = True

                    elif len(bef_words) == 2:
                        b_check = re.findall("NN.*\sVB[DNP]$", b_tags)
                        a_check = re.findall("NN|JJ|VB[DN]?$|CD", aft_tags)
                        bef_sent = " ".join((bef_words[-2:]))
                        if b_check and not a_check and any(w_b in bef_words for w_b in bef_existed):
                            recommed_check = True
            elif tags[w_i] == "JJ":
                if token.endswith("ed") or token.endswith("ing"):
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
                            b_check = re.findall("VB[DNP][\sRB]*$", b_tags)
                            b_and_check = re.findall("VB[DNP]\sCC\s[RB]*$|RB$", b_tags)
                            a_check = re.findall("NN|JJ|VB[DN]?$|CD", aft_tags)
                            if b_check and not a_check and any(w_b in bef_words for w_b in
                                                               bef_existed):
                                recommed_check = True
                            if b_and_check and not a_check:
                                recommed_check = True
                        elif len(bef_words) == 2:
                            b_check = re.findall("NN.*\sVB[DNP]$|PRP\sVB[DNP]$", b_tags)
                            a_check = re.findall("NN|JJ|VB[DN]$|CD", aft_tags)
                            if b_check and not a_check and any(w_b in bef_words for w_b in bef_existed):
                                recommed_check = True
            elif tags[w_i] == "VBG":
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
                    if tags_b_words:
                        b_check_2 = re.findall("^VB[DN]\sIN$", tags_b_words)
                        b_check_3 = re.findall("IN$|CC$", tags_b_words)
                    elif bef_tags_1:
                        b_check_1 = re.findall("IN", bef_tags_1)
                    if aft_tags:
                        a_check = re.findall("JJ|VB[DN]$|CD", aft_tags)
                        a_check_3 = re.findall("IN", aft_tags)
                    if len(bef_words) == 1:
                        if not a_check and b_check_1:
                            recommed_check = True
                    if len(bef_words) == 2:
                        if not a_check and b_check_2:
                            recommed_check = True
                        if b_check_3 and a_check_3:
                            recommed_check = True
            elif nn_check and token.endswith("ing"):
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
                if bef_words and aft_words:
                    if tags_b_words:
                        b_check_2 = re.findall("^VB[DN]\sIN$|CC$|RB$", tags_b_words)
                    elif bef_tags_1:
                        b_check_1 = re.findall("IN", bef_tags_1)
                    if aft_tags:
                        a_check = re.findall("NN|JJ|VB[DN]?$|CD", aft_tags)
                    if len(bef_words) == 1:
                        if not a_check and b_check_1:
                            recommed_check = True
                    if len(bef_words) == 2:
                        if not a_check and b_check_2:
                            recommed_check = True

    return recommed_check, nn_check

def sim_filter(word2vec_path, fasttext_path,seeds, recommed_words, max_iterations, sentences, sim_level, freq_threshold, bef_existed,
               sentences_tag, log_path, verb_lemma):
    model = word2vec.Word2Vec.load(word2vec_path)
    rejected_words = list()
    all_recommed_results = list()
    recommed_results = dict()
    not_in_vocab = list()
    new_recoms = list()
    for r_word in recommed_words:
        check = re.findall(r"^[a-zA-Z\-\/]+$", r_word)
        if check:
            new_recoms.append(r_word)
    for iter_n in range(max_iterations):
        all_sims = list()
        fasttext_to_words = list()
        for seed in tqdm(seeds):
            sims_unique_words = list()
            if seed in model.wv.index2word:
                seed_sims = word2vec_sim_filter(model, seed, sim_level)
                for w_s in seed_sims:
                    word_s = w_s.replace(",", "")
                    word_s = word_s.replace(".", "")
                    word_s = word_s.replace(")", "")
                    word_s = word_s.replace("(", "")
                    word_s = word_s.replace(";", "")
                    word_s = word_s.lower()
                    check = re.findall(r"^[a-zA-Z\-]+$", word_s)
                    if check and word_s not in seeds and word_s in new_recoms and word_s not in sims_unique_words:  # and word_s not in sims_unique_words
                        all_sims.append(word_s)
                        sims_unique_words.append(word_s)
            else:
                not_in_vocab.append(seed)
            to_words = fasttext_sims(fasttext_path, seed, new_recoms, 0.5, seeds)

            fasttext_to_words.extend(to_words)
        fasttext_to_words = list(set(fasttext_to_words))
        fasttext_recoms = nn_lem_word(seeds, fasttext_to_words, verb_lemma)  # Lemmatization for verbs such as VBG/VBN, but does not work well for nouns
        input_words = list()
        if len(fasttext_recoms) > 0:
            for word_g in tqdm(fasttext_recoms):
                for sent in sentences:
                    if word_g in sent:
                        vb_check = fasttext_not_vb_check(word_g, sent, sentences_tag)
                        if vb_check:
                            input_words.append(word_g)
                            break
        count_dict = dict()
        unique_sp = set(all_sims)
        for u_s in unique_sp:
            count_dict[u_s] = countX(all_sims, u_s)
        generated_words = list()
        for word, fre in count_dict.items():
            if fre >= freq_threshold and word not in seeds:
                generated_words.append(word)
        generated_words = reject_filter(generated_words, rejected_words)
        generated_words = list(set(generated_words))
        if len(generated_words) > 0:
            nn_list = list()
            for word_g in tqdm(generated_words):
                for sent in sentences:
                    if word_g in sent:
                        check_g, nn_check = chunk_window(word_g, sent, bef_existed,
                                                         sentences_tag)  # Verify that the word matches the POS we need in a given sentence
                        if check_g:
                            input_words.append(word_g)

            nn_list = list(set(nn_list))
            nn_token = nn_lem_word(seeds, nn_list, verb_lemma)
            if nn_token:
                input_words.extend(nn_token)
            if rejected_words:
                for r_w in rejected_words:
                    if r_w in input_words:
                        input_words.remove(r_w)

            input_words = list(set(input_words))
            print("Recommend words number in this iteration: %d" % (len(input_words)))
            with open(os.path.join(
                    log_path,
                    "pre-" + str(iter_n) + ".txt"), encoding="utf-8", mode="w") as f:
                f.write(str(input_words))
            with open(os.path.join(
                    log_path,
                    str(iter_n) + ".txt"), encoding="utf-8", mode="w+") as f:
                f.write(str(input_words) + "\n")
                f.write(str(len(input_words)) + "\n")
            r_word = input("Please input rejected words by check %s" % str(input_words))
            rejected_words.extend(eval(r_word))
            for b_w in eval(r_word):
                if b_w in input_words:
                    input_words.remove(b_w)
            with open(os.path.join(
                    log_path,
                    str(iter_n) + ".txt"), encoding="utf-8", mode="a") as f:
                f.write(str(input_words) + "\n")
                f.write(str(len(input_words)))
            seeds.extend(input_words)
            recommed_results[iter_n] = input_words
            all_recommed_results.extend(input_words)
            print("The %d iteration recommened %d entities" % (iter_n, len(input_words)))
            if len(input_words) == 0:
                print("Iteration ending.")
                break
        if len(generated_words) == 0:
            print("Iteration ending.")
            break

    return recommed_results, not_in_vocab, all_recommed_results


def fasttext_filter(words_now, recommed_words, model_path, sim_treshold):
    model = fasttext.load_model(model_path)
    recommed_results = list()
    for wr in tqdm(recommed_words):
        wv_2 = model.get_word_vector(wr)
        n_recom = 0
        for wn in words_now:
            wv_1 = model.get_word_vector(wn)
            sim = cosine_similarity(wv_1.reshape(1, -1), wv_2.reshape(1, -1))
            if sim > sim_treshold:
                n_recom += 1
        if n_recom > len(words_now) / 5:
            recommed_results.append(wr)
    return recommed_results

def word_lemma(words):
    wnl = WordNetLemmatizer()
    verb_lemma = dict()
    for word in words:
        verb_lemma[word] = list()
        verb_lemma[word].append(wnl.lemmatize('quenching', 'n'))
        verb_lemma[word].append(wnl.lemmatize('quenching', 'v'))
    return verb_lemma

def candidate_verb(corpus_path):
    with open(corpus_path, 'r', encoding=("utf-8")) as file_lab:
        sent_info = file_lab.readlines()
        all_verb = list()
        for sent in tqdm(sent_info):
            words = nltk.word_tokenize(sent)
            sent = " ".join(words)
            tagged_sentence = nltk.tag(sent)
            for word in tagged_sentence:
                check_w = re.findall("^VB[BN]?", word[1])
                check_t = re.findall("^[A-Za-z\-]+$", word[0])
                if check_w and check_t and word not in stopword_list:
                    l_w = word[0]
                    all_verb.append(l_w.lower())
                check_jj = re.findall("JJ", word[1])
                check_nn = re.findall("NN", word[1])
                if check_jj and "-" in word[0] and check_t:
                    all_verb.append(word[0].lower())
                if check_nn and "-" in word[0] and check_t:
                    all_verb.append(word[0].lower())
    all_verb = list(set(all_verb))
    return all_verb

bef_existed = ["was", "were", "been", "is", "are"]
corpus_path = r".../all_corpus.txt"
verb_data = candidate_verb(corpus_path)
verb_lemma = word_lemma(verb_data)

with open(corpus_path, mode='r', encoding='utf8') as fp:
    sentences = fp.readlines()

sentences_tag = dict()
for sent in tqdm(sentences):
    sentences_tag[sent] = nltk.tag(sent)

start_seeds = []
word2vec_path = r".../model/ft_4.bin"
fasttex_path =  r".../model/ft_2.bin"
log_p = r".../log"

recommed_results, not_in_vocab, all_recommed_results = sim_filter(word2vec_path=word2vec_path, fasttext_path=fasttex_path, seeds=start_seeds,
                                                                  recommed_words=verb_data, max_iterations=40,
                                                                  sim_level=0.46, freq_threshold=2, sentences=sentences.lower(),
                                                                  bef_existed=bef_existed, sentences_tag=sentences_tag,
                                                                  log_path=log_p,verb_lemma=verb_lemma)

with open(r'.../results.json','w', encoding='utf8')as fo:
    fo.write(str(all_recommed_results))


