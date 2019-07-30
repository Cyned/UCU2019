import numpy as np

from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import Tokenizer


class Doc2Vec(TransformerMixin):
    def __init__(self, word2vec, tokenizer = Tokenizer(stem='lem', remove_spec=True).tokenize):
        """
        :param word2vec: word2vec model
        :param tokenizer: tokenizer to preprocess text
        """
        self.word2vec = word2vec
        self.tokenize = tokenizer

        self.tfidf = TfidfVectorizer(
            lowercase    = True,
            tokenizer    = self.tokenize,
            analyzer     = 'word',
            ngram_range  = (1, 1),
            max_df       = 1.0,
            min_df       = 1,
            max_features = None,
        )
        self.feature_names = []

    def fit(self, x_train, y_train, *args, **kwargs):
        # self.tfidf.fit(x_train, y_train)
        # self.feature_names = self.tfidf.get_feature_names()
        return self

    def doc2vec(self, text: str) -> np.array:
        """
        Get vector of the text
        :param text: text to get vector from
        :return: vector of the text
        """
        # tfidf_matrix = self.tfidf.transform([text])
        # vectors = []
        # for token in self.tokenize(text):
        #     if token in self.word2vec and token in self.feature_names:
        #         tfidf_score = tfidf_matrix[0, self.feature_names.index(token)]
        #         vectors.append(self.word2vec[token] * tfidf_score)
        vectors = [self.word2vec[token] for token in self.tokenize(text) if token in self.word2vec]
        if not vectors:
            return np.zeros(300)
        return np.mean(vectors, axis=0)

    def transform(self, x_train):
        return np.array([self.doc2vec(text=text) for text in x_train])
