from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from normalizer import Normalizer, POSTagNormalizer
from lemmatizer import Lemmatizer, POSTagLemmatizer


class PreProcessor:
    def __init__(self, pos_tagging=False):
        self.pos_tagging = pos_tagging
        self.normalizer = Normalizer() if not pos_tagging else POSTagNormalizer()
        self.lemmatizer = Lemmatizer() if not pos_tagging else POSTagLemmatizer()

    @classmethod
    def tokenize(cls, content):
        sents = sent_tokenize(content)
        return [word_tokenize(sent) for sent in sents]

    def tag(self, sents):
        return [self.tag(sent) for sent in sents] if self.pos_tagging else sents

    def process(self, content):
        tokenized = self.tag(self.tokenize(content))
        normalized = self.normalizer.normalize(tokenized)
        return self.lemmatizer.lemmatize(normalized)
