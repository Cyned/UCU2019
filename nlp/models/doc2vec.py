from sklearn.base import TransformerMixin

from gensim.models.keyedvectors import KeyedVectors


class Doc2Vec(TransformerMixin):
    def __init__(self):
        self.word2vec = KeyedVectors.load_word2vec_format("wiki-news-300d-1M.vec")


