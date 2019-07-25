import nltk

import pandas as pd

from os.path import join as path_join

from timeit_ import timeit_context
from config import DATA_DIR


def load_original_data() -> pd.DataFrame:
    nltk.download('movie_reviews')
    from nltk.corpus import movie_reviews

    # Extract lists of ids of the reviews that are labeled as negative and positive
    negative_ids = movie_reviews.fileids('neg')
    positive_ids = movie_reviews.fileids('pos')

    # Let's prepare the list of texts and their classes as a training examples
    # Note: the texts are already preprocessed and split to tokens.
    negative_reviews = [
      " ".join(movie_reviews.words(fileids=[f]))
        for f in negative_ids
    ]
    positive_reviews = [
        " ".join(movie_reviews.words(fileids=[f]))
        for f in positive_ids
    ]

    texts  = negative_reviews + positive_reviews
    labels = [0] * len(negative_reviews) + [1] * len(positive_reviews)
    return pd.DataFrame({'text': texts, 'target': labels})


if __name__ == '__main__':
    with timeit_context('Read data'):
        data = load_original_data()
    with timeit_context('Write data to parquet'):
        data.to_parquet(path_join(DATA_DIR, 'movie_reviews.parquet'))
