import logging

from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


class BlendingFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, models: List[BaseEstimator]):
        """
        :param models: chain of the models to use in blending
        """
        self.models = models

    def fit(self, X, y):
        """
        Fit the model
        :param X: x train values
        :param y: target values
        """
        for model in self.models:
            model.fit(X, y)
        return self

    def transform(self, X):
        predictions = []
        for model in self.models:
            try:
                predictions.append(model.predict_proba(X)[:, 1])
            except Exception as e:
                logging.warning(e)
                predictions.append(model.predict(X))
        return list(zip(*predictions))
