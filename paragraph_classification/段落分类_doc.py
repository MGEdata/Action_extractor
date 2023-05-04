# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 22:48:35 2019

@author: 35732
"""
from gensim.models import word2vec
import os
import nltk
import re
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from tqdm import tqdm
from sklearn import utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from gensim.models import Doc2Vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
import numpy as np

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            # 这里原先将长度小于2的字符串删除了
            # if len(word) < 2:
            #     continue
            tokens.append(word)
    return tokens

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

def get_vectors(model, tagged_docs, doc_label):
    sents = tagged_docs.values
    outcome = list()
    # infer_vec = model.infer_vector(doc.words, steps=5)
    for doc in sents:
        doc_text = " ".join(doc.words)
        model_infer = model.infer_vector(doc.words, steps=20)
        # if "experiment" in doc_label[doc_text].lower() or "method" in doc_label[doc_text].lower():
        #     tag_vec = np.ones(1)
        #     model_i = np.concatenate((model_infer,tag_vec),axis=0)
        # else:
        #     tag_vec = np.zeros(1)
        #     model_i = np.concatenate((model_infer,tag_vec),axis=0)
        outcome.append((doc.tags[0], model_infer))
    targets, regressors = zip(*outcome)
    # targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

def get_accuracy_f1(lable_path):
    csv_data = pd.read_csv(lable_path)
    df = pd.DataFrame(csv_data)
    df_shape = df.shape
    length = df_shape[0]
    outcome={}
    for i in range(length):
        example_i = df.iloc[i]
        if len(example_i) > 0:
            outcome.update({example_i[0]:example_i[1]})
    paragraph_list = list(outcome.keys())
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    words_list=[]
    outcome_1={}
    doc_label = dict() #用于记录每一个自然段对应的标签
    for i in range(len(paragraph_list)):
        r='[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
        paragraph_0=re.sub(r,' ',paragraph_list[i].lower())
        words_list_0 = nltk.word_tokenize(paragraph_0)
        words_list.append(words_list_0)
        keys = " ".join(words_list[i])
        outcome_1.update({keys:df.iloc[i][1]})
        doc_label.update({keys:df.iloc[i][2]})
    # 可以去重
    df = pd.DataFrame(pd.Series(outcome_1), columns=['y'])   #将字典转化为dataframe的形式,第一列会变成索引列
    df = df.reset_index().rename(columns={'index':'x'})
    train, test = train_test_split(df, test_size=0.2, random_state=10)    #分为训练集和测试集
    train_tagged = train.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['x']), tags=[r.y]), axis=1)
    test_tagged = test.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['x']), tags=[r.y]), axis=1)
    cores = multiprocessing.cpu_count()
    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
    model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

    # 以下进行
    for epoch in range(40):
        model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
        model_dbow.alpha -= 0.002   #学习率
        model_dbow.min_alpha = model_dbow.alpha

    model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
    model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])
    for epoch in range(30):
        model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)#打乱各个自然段的顺序
        model_dmm.alpha -= 0.002
        model_dmm.min_alpha = model_dmm.alpha
    new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])
    y_train, X_train = get_vectors(new_model, train_tagged, doc_label)
    y_test, X_test = get_vectors(new_model, test_tagged, doc_label)
    Log = LogisticRegression(max_iter=1000,class_weight='balanced').fit(X_train, y_train)
    y_pred = Log.predict(X_test)
    print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
    print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

get_accuracy_f1(r"E:/文本挖掘/工作三-工艺抽取/段落分类/101段落标记分类/add_corpus/2rd_add_shuff.csv")


#y_train, X_train = vec_for_learning(model_dbow, train_tagged)
#y_test, X_test = vec_for_learning(model_dbow, test_tagged)
#logreg = LogisticRegression(max_iter=1000, n_jobs=1, C=1e5)
#logreg.fit(X_train, y_train)
#y_pred = logreg.predict(X_test)






