import nltk
import re

from nltk.corpus import stopwords as stw

from typing import List, Set
from models.rule_based import get_dictionaries


stopwords        = stw.words('english')
stemmer          = nltk.PorterStemmer()
lemmatizer       = nltk.WordNetLemmatizer()
lemmatizer.stem  = lemmatizer.lemmatize
stemmer.old_stem = stemmer.stem
stemmer.stem     = lambda word: stemmer.old_stem(lemmatizer.lemmatize(word))


class Tokenizer(object):
    """ Custom Tokenizer """

    no_suffix = ('no', 'not')
    spec_symbols = [f' {spec} ' for spec in [
        r'`', r'~', r'!', r'@', r'#', r'\$', r'%', r'\^',
        r'&', r'\*', r'\(', r'\)', r'_', r'-', r'+', r'=',
        r'\{', r'\[', r'\]', r'\}', r'\|', r'\\',
        r':', r';', r'"', r'<', r',', r'>', r'\.', r'\?', r'/', r"'",
    ]]

    def __init__(self, stem: str = None, splitter = None, remove_spec: bool = False):
        """

        :param stem: `lem` or `stem`
        :param splitter:
        :param remove_spec:
        """
        if stem == 'lem':
            self.stemmer = lemmatizer
        elif stem == 'stem':
            self.stemmer = stemmer
        else:
            self.stemmer = None

        if not splitter:
            self.splitter = lambda x: x.split()
        else:
            self.splitter = splitter

        self.remove_spec_cond = remove_spec

    def remove_mentions(self, text: str) -> str:
        """ Remove mentions """
        return re.sub(r"@[^:| ]+:? ?", '', text)

    def remove_hashtags(self, text: str) -> str:
        """ Remove hash tags """
        return re.sub(r"#+\w* ?", '', text)

    def remove_digits(self, text: str) -> str:
        """ Remove all numbers """
        return re.sub(r'[-+]?\d+(\.[0-9]*)?', '', text)

    def remove_spec(self, text: str) -> str:
        """ Remove special symbols """
        for sym in self.spec_symbols:
            text = re.sub(sym, ' ', text)
        return text

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """ Remove stopwords """
        return [token for token in tokens if token not in stopwords]

    def make_negative_common(self, text: str) -> str:
        """ Union negative words to one word via `_` """
        while True:
            match = re.search(f"({'|'.join(self.no_suffix)}) \w*", text)
            if not match:
                break
            new_sub_text = 'not_' + re.sub(f"({'|'.join(self.no_suffix)})", '', text[match.start(): match.end() + 1])
            text = text[:match.start()] + new_sub_text + text[match.end() + 1:]
        return text

    def tokenize(self, text: str) -> List[str]:
        """ Tokenize text """
        text = text.lower()
        text = self.remove_mentions(text=text)
        text = self.remove_digits(text=text)
        text = self.remove_hashtags(text=text)

        if self.remove_spec_cond:
            text = self.remove_spec(text=text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        tokens = self.splitter(text)
        tokens = self.remove_stopwords(tokens=tokens)
        if self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens


class DictTokenizer(object):
    def __init__(self, splitter = None):
        positive, negative = get_dictionaries()
        self.dict = {word for word in positive + negative + ['no', 'not'] if len(word.split()) == 1}
        if not splitter:
            self.splitter = lambda x: x.split()

    def tokenize(self, text: str) -> Set[str]:
        text = text.lower()
        return {lemmatizer.lemmatize(word) for word in self.splitter(text)} & self.dict


if __name__ == '__main__':
    text = 'I do not care about this stuff. #Please remove 123 !'
    tokenizer = Tokenizer(stem='stem', splitter=None, remove_spec=True)
    tokens = tokenizer.tokenize(text=text)
    print(tokens)
    # =================================================================
    text = 'I do not care about this stuff. #Please remove 123 !'
    tokenizer = DictTokenizer(splitter=None)
    tokens = tokenizer.tokenize(text=text)
    print(tokens)

