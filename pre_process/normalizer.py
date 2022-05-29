import string

import nltk


class Normalizer:
    def __init__(
            self,
            min_len: int = 3,
            lower_cased: bool = True,
            stopword_removal: bool = True,
            stopwords_domain: list = None,
            punctuation_removal: bool = True
    ):
        if stopwords_domain is None:
            stopwords_domain = list()
        self.min_len = min_len
        self.lower_cased = lower_cased
        self.stopword_removal = stopword_removal
        self.stopwords_domain = stopwords_domain + nltk.corpus.stopwords.words('english')
        self.punctuation_removal = punctuation_removal

    def remove_stopwords(self, sentences):
        stopwords = [x.lower() for x in self.stopwords_domain]
        return [[word for word in sentence if (word.lower() not in stopwords)]
                for sentence in sentences]

    def remove_punctuations(self, sentences):
        return [[word for word in sentence if word not in string.punctuation]
                for sentence in sentences]

    def lower_case(self, sentences):
        return [[word.lower() for word in sentence if len(word) > self.min_len]
                for sentence in sentences]

    def filter_min_len(self, sentences):
        return [[word for word in sentence if len(word) > self.min_len]
                for sentence in sentences]

    def normalize(self, tokenized_sentences):
        sentences = tokenized_sentences
        if self.stopword_removal:
            sentences = self.remove_stopwords(sentences)
        if self.punctuation_removal:
            sentences = self.remove_punctuations(sentences)
        if self.lower_cased:
            sentences = self.lower_case(sentences)
        elif self.min_len > 1:
            sentences = self.filter_min_len(sentences)
        return sentences


class POSTagNormalizer(Normalizer):
    """data normalizer with pos-tag"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def remove_stopwords(self, sentences):
        stopwords = [x.lower() for x in self.stopwords_domain]
        return [[(word, pos) for word, pos in sentence if (word.lower() not in stopwords)]
                for sentence in sentences]

    def remove_punctuations(self, sentences):
        return [[(word, pos) for word, pos in sentence if word not in string.punctuation]
                for sentence in sentences]

    def lower_case(self, sentences):
        return [[(word.lower(), pos) for word, pos in sentence if len(word) > self.min_len]
                for sentence in sentences]

    def filter_min_len(self, sentences):
        return [[(word, pos) for word, pos in sentence if len(word) > self.min_len]
                for sentence in sentences]
