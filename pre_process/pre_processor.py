from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize

from pre_process.lemmatizer import Lemmatizer, POSTagLemmatizer
from pre_process.normalizer import Normalizer, POSTagNormalizer


class PreProcessor:
    def __init__(self, pos_tagging=False):
        self.pos_tagging = pos_tagging
        self.normalizer = Normalizer() if not pos_tagging else POSTagNormalizer()
        self.lemmatizer = Lemmatizer() if not pos_tagging else POSTagLemmatizer()

    @classmethod
    def tokenize(cls, content):
        sentences = sent_tokenize(content)
        return [word_tokenize(sentence) for sentence in sentences]

    def tag(self, sentences):
        return [pos_tag(sentence) for sentence in sentences] if self.pos_tagging else sentences

    def process(self, content):
        tokenized = self.tag(self.tokenize(content))
        normalized = self.normalizer.normalize(tokenized)
        return self.lemmatizer.lemmatize(normalized)
