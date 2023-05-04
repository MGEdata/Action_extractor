

from gensim.models import word2vec
import logging


sentences = word2vec.Text8Corpus(r'.\all_text.txt') #经过句子开头字母小写处理的语料
model = word2vec.Word2Vec(sentences, size=100, hs=1, min_count=1, window=5)
model.save(r'.\ft_3.bin')

import fasttext
from sklearn.metrics.pairwise import cosine_similarity

# model的训练代码
text_path = r"E:\fasttext\model\ft_2.bin"
model = fasttext.train_unsupervised(input=r'.\all_text.txt',model="skipgram", lr=0.05, dim=100,epoch=5, minCount=5, minn=3, maxn=8, wordNgrams=4,loss="ns")
model.save_model(text_path)

# model的调用代码
# model = fasttext.load_model(r"E:\fasttext\model\ft_1.bin")
# wv_1 = model.get_word_vector("sample").reshape(1,-1)
# wv_2 = model.get_word_vector("specimen").reshape(1,-1)
# sim = cosine_similarity(wv_1, wv_2)