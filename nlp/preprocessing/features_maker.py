import re

import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin


class FeaturesMaker(TransformerMixin):

    spec_symbols = [f' {spec} ' for spec in [
        r'`', r'~', r'!', r'@', r'#', r'\$', r'%', r'\^',
        r'&', r'\*', r'\(', r'\)', r'_', r'-', r'+', r'=',
        r'\{', r'\[', r'\]', r'\}', r'\|', r'\\',
        r':', r';', r'"', r'<', r',', r'>', r'\.', r'\?', r'/', r"'",
    ]]

    def __init__(self):
        pass

    def fit(self, x_train, y_train):
        return self

    def transform(self, x_train):
        return pd.DataFrame([
            [
                self.count_no(text=sample),
                self.count_commas(text=sample),
                self.count_exclamatory(text=sample),
                self.count_numbers(text=sample),
                self.count_questions(text=sample),
                self.count_sentences(text=sample),
                self.count_unique_words(text=sample)
            ]
            for sample in x_train
        ], columns=['no', 'commas', 'exclamatory', 'numbers', 'questions', 'sentences', 'unique_words'],
        )

    @staticmethod
    def count_no(text: str) -> int:
        """

        :param text:
        :return:
        """
        return len(re.findall(r"(no)|(not)|(n't)", text))

    @staticmethod
    def count_exclamatory(text: str) -> int:
        """

        :param text:
        :return:
        """
        return text.count('!')

    @staticmethod
    def count_questions(text: str) -> int:
        """

        :param text:
        :return:
        """
        return text.count('?')

    @staticmethod
    def count_numbers(text: str) -> int:
        """

        :param text:
        :return:
        """
        return len(re.findall(r'[0-9]', text))

    @staticmethod
    def count_sentences(text: str) -> int:
        return len(re.findall(r'[\?!\.;]+ ?', text))

    @staticmethod
    def count_commas(text: str) -> int:
        """

        :param text:
        :return:
        """
        return text.count(',')

    @staticmethod
    def sentence_mean(text: str) -> float:
        """

        :param text:
        :return:
        """
        sentences = re.split(r'[\?!\.;]+ ?', text)[:-1]
        return np.mean([len(sentence.split(' ')) for sentence in sentences], axis=0)

    def count_unique_words(self, text: str) -> int:
        """

        :param text:
        :return:
        """
        for sym in self.spec_symbols:
            text = re.sub(sym, ' ', text)
        return len(set(text.split(' ')))


if __name__ == '__main__':
    fm = FeaturesMaker()
    text = "Hello, My name is GGG. I am not girl, but boy! Don't touch me please!"
    result = fm.transform([text])
    print(result)