import re

import numpy as np

from sklearn.base import BaseEstimator
from typing import List, Iterable
from sklearn.metrics import accuracy_score

from preprocessing import POSITIVE, NEGATIVE


class RuleBased(BaseEstimator):

    def __init__(self, eval_metric = accuracy_score):
        self.positive = POSITIVE
        self.negative = NEGATIVE

        self.eval_metric = eval_metric
        self.threshold   = None

    def fit(self, x_train, y_train):
        plain_text = ' '.join(x_train)

        predictions  = self.predict_proba(x_train)
        eval_metrics = []
        thrs         = []
        for thr in np.arange(0.05, 0.9, 0.05):
            eval_metrics.append(self.eval_metric(predictions > thr, y_train))
            thrs.append(thr)
        self.threshold = thrs[np.argmax(eval_metrics)]
        return self

    def get_tone(self, text: str, debug: bool = False):
        """

        :param text:
        :param debug:
        :return:
        """
        text = text.lower()

        pos  = len([1 for word in POSITIVE if word in text])
        neg  = len([1 for word in NEGATIVE if word in text])
        neg += len(re.findall(r"(do not)|(don't)|(does not)|(doesn't)|(didn't)|(did not)|(has not)|(hasn't)|(have not)|(haven't) like", text))

        if debug:
            print(f'POS: {pos}, NEG: {neg}')
        # return (pos - neg) / (max([pos, neg]))
        return pos / (pos + neg)

    def predict_proba(self, x_test: Iterable[str], debug: bool = False) -> np.array:
        """

        :param x_test:
        :param debug:
        :return:
        """
        return np.array([self.get_tone(text=text, debug=debug) for text in x_test])

    def predict(self, *args, **kwargs):
        return [i > self.threshold for i in self.predict_proba(*args, **kwargs)]

    def transform(self, x_test: Iterable[str]):
        """

        :param x_test:
        :return:
        """
        return [[self.count_negative([text])[0], self.count_positive([text])[0]] for text in x_test]

    def count_positive(self, x_test: Iterable[str]) -> List[int]:
        """

        :param x_test:
        :return:
        """
        return [self.count_words(text=text, vocab=POSITIVE) for text in x_test]

    def count_negative(self, x_test: Iterable[str]) -> List[int]:
        """

        :param x_test:
        :return:
        """
        return [self.count_words(text=text, vocab=NEGATIVE) for text in x_test]

    def count_words(self, text: str, vocab: List[str]) -> int:
        """

        :param text:
        :param vocab:
        :return:
        """
        return np.sum([word in text for word in vocab])


if __name__ == '__main__':
    model = RuleBased()
    result = model.predict_proba(['I do not like this movie at all. It was awful and not interesting'], debug=True)
    print(result)
