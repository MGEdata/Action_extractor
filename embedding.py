

from gensim.models import word2vec
import logging
from chemdataextractor.doc import Paragraph



with open(r"./ori.txt", "r",encoding="utf-8",) as ori_file:
    all_paragraphs = ori_file.read()
paras = Paragraph(all_paragraphs)
sents = paras.sentences
new_sents = ""
for sent in sents:
    new_sents += sent.lower()
with open(r"./reorg.txt", "w+",encoding="utf-8",) as re_file:
    re_file.write(new_sents)

sentences = word2vec.Text8Corpus(r"./reorg.txt") #经过句子开头字母小写处理的语料
model = word2vec.Word2Vec(sentences, size=100, hs=1, min_count=1, window=5)
model.save(r'.\ft_3.bin')

import fasttext
from sklearn.metrics.pairwise import cosine_similarity

# model的训练代码
text_path = r".\ft_2.bin"
model = fasttext.train_unsupervised(input=r"./reorg.txt",model="skipgram", lr=0.05, dim=100,epoch=5, minCount=5, minn=3, maxn=8, wordNgrams=4,loss="ns")
model.save_model(text_path)

# model的调用代码
# model = fasttext.load_model(r"E:\fasttext\model\ft_1.bin")
# wv_1 = model.get_word_vector("sample").reshape(1,-1)
# wv_2 = model.get_word_vector("specimen").reshape(1,-1)
# sim = cosine_similarity(wv_1, wv_2)