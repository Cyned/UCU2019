import pandas as pd

from os.path import join as path_join
from typing import Callable
from sklearn.model_selection import cross_val_score, KFold
from sklearn.base import BaseEstimator

from config import DATA_DIR


def custom_metric() -> Callable:
    """ Returns metric function """
    def closure(classifier: BaseEstimator, verbose: int = 0, n_jobs: int = 1) -> float:
        """
        Count accuracy by cross validation (split=5) on movie_reviews corpora.
        :param classifier: model that has `fit`, `predict` methods. An input for the model must be the plain text.
        :param verbose: level of logging.
	:param n_jobs: number of workers
        :return: accuracy of the model
        """
        result = cross_val_score(
            estimator = classifier,
            X         = data['text'].values,
            y         = data['target'].values,
            scoring   = 'accuracy',
            cv        = kfold,
            n_jobs    = n_jobs,
            verbose   = verbose,
        ).mean()
        return result
    random_state = 743662043  # Just nobody could repeat it!
    data = pd.read_parquet(path_join(DATA_DIR, 'movie_reviews.parquet'))  # to keep the test data untouched!
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
    return closure
