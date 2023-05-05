# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:23:59 2021

@author: win
"""

import nltk
import re
from dictionary import Dictionary

error_chunks = Dictionary(r"dictionary.ini").error_chunks

def intensified_tokenize(text):
    
    # few angstroms in diameter . 2 Experimental details 2.1 Materi
    conds = re.findall("[^\.]{,10}[a-z]{2,}\.\s[0-9]+\s[A-Z][^\.]{,10}", text)
    for cond in conds:
        rep = re.sub("\.\s","\.\n",cond)
        text = text.replace(cond,rep,1)
        
    conds = re.findall("[^\.]{,10}[A-Z][a-z]{2,}\.\s+[0-9][^\.]{,10}", text)
    for cond in conds:
        rep = re.sub("\.\s+",".",cond)
        text = text.replace(cond,rep,1)
    conds_2 = re.findall("[^\.]{10}[A-Za-z]\.\s[A-Z][^\.]{10}", text)
    for cond in conds_2:
        rep = re.sub("\.\s+",".\n",cond)
        text = text.replace(cond,rep,1)
    sent_tokenize = nltk.sent_tokenize(text)
    all_sents = list()
    for sent in sent_tokenize:
        sub_sents = sent.split("\n")
        if sub_sents:
            for st in sub_sents:
                all_sents.append(st)
        else:
            all_sents.append(sent)

    return all_sents

def sent_process(sent_i,sents,modified_sents,output_sents,error_chunks):

    a_sent = sents[sent_i].strip()

    next_judge = None
    if a_sent[-1] != "." and sents[sent_i+1][0].islower():
        new_sent = sents[sent_i] + ' ' + sents[sent_i+1]
        output_sents.append(new_sent)
        del sents[sent_i+1]
        sents[sent_i] = new_sent
        modified_sents += 1
        if sents[sent_i][-1] != "." and sents[sent_i+1][0].islower():
            next_judge = True
    
    elif a_sent[-1] == "." and any(list(a_sent.endswith(chunk) for chunk in error_chunks)):
        new_sent = sents[sent_i] + ' ' +sents[sent_i+1]
        output_sents.append(new_sent)
        del sents[sent_i+1]
        sents[sent_i] = new_sent
        modified_sents += 1
        if sents[sent_i][-1] == "." and any(list(a_sent.endswith(chunk) for chunk in error_chunks)):
            next_judge = True
    else:
        if output_sents and output_sents[-1] != a_sent.strip():
            output_sents.append(a_sent.strip())
        if not output_sents:
            output_sents.append(a_sent.strip())

    return next_judge,sents,modified_sents,output_sents

def sents_reorg(sents, error_chunks):

    """When a sentence does not end with a period and the next sentence does not begin with a capital letter, the two sentences are spliced"""
    """When a sentence ends with a period and ends with the word in error_chunks, the two sentences are spliced"""

    modified_sents = 0
    output_sents = list()
    try:
        for sent_i in range(len(sents)):
                try:
                    for num in range(5):
                        next_judge,sents,modified_sents,output_sents = sent_process(sent_i,sents,modified_sents,
                                                                                    output_sents,error_chunks)
                        if not next_judge:
                            break
                        else:
                            del output_sents[-1]
                except Exception as e:
                    print(e)
                    print(sents[sent_i])
                output_len = len(output_sents)
    except Exception as e:
        print(e)

    return output_sents

