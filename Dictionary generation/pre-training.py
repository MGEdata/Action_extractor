import fasttext
from gensim.models import word2vec

text_path = r".\fasttext.bin"
model = fasttext.train_unsupervised(input=r'.\reorg_corpus.txt',model="skipgram", lr=0.05, dim=100,epoch=5, minCount=5, minn=3, maxn=8, wordNgrams=4,loss="ns")
model.save_model(text_path)

sentences = word2vec.Text8Corpus(r'.\reorg_corpus.txt')
model = word2vec.Word2Vec(sentences, size=100, hs=1, min_count=1, window=10)
model.save(r'.\word2vec.bin')