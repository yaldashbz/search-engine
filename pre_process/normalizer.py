import nltk
import string


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

    def remove_stopwords(self, sents):
        stopwords = [x.lower() for x in self.stopwords_domain]
        return [[word for word in sentence if (word.lower() not in stopwords)]
                for sentence in sents]

    def remove_punctuations(self, sents):
        return [[word for word in sentence if word not in string.punctuation]
                for sentence in sents]

    def lower_case(self, sents):
        return [[word.lower() for word in sentence if len(word) > self.min_len]
                for sentence in sents]

    def filter_min_len(self, sents):
        return [[word for word in sentence if len(word) > self.min_len]
                for sentence in sents]

    def normalize(self, tokenized_sents):
        sents = tokenized_sents
        if self.stopword_removal:
            sents = self.remove_stopwords(sents)
        if self.punctuation_removal:
            sents = self.remove_punctuations(sents)
        if self.lower_cased:
            sents = self.lower_case(sents)
        elif self.min_len > 1:
            sents = self.filter_min_len(sents)
        return sents


class POSTagNormalizer(Normalizer):
    """data normalizer with pos tag"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def remove_stopwords(self, sents):
        stopwords = [x.lower() for x in self.stopwords_domain]
        return [[(word, pos) for word, pos in sentence if (word.lower() not in stopwords)]
                for sentence in sents]

    def remove_punctuations(self, sents):
        return [[(word, pos) for word, pos in sentence if word not in string.punctuation]
                for sentence in sents]

    def lower_case(self, sents):
        return [[(word.lower(), pos) for word, pos in sentence if len(word) > self.min_len]
                for sentence in sents]

    def filter_min_len(self, sents):
        return [[(word, pos) for word, pos in sentence if len(word) > self.min_len]
                for sentence in sents]
