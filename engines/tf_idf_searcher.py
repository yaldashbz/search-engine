import itertools
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from engines.base_searcher import BaseSearcher
from engines.utils import cosine_sim, DataOut
from retriever.utils import get_contents


class TFIDFSearcher(BaseSearcher):
    def __init__(self, data):
        super().__init__(data)
        contents = get_contents(data)

        self.tfidf = self._get_tfidf()
        self.matrix = self.tfidf.fit_transform(contents)
        self.vocabulary = self.tfidf.get_feature_names_out()
        self.output_cls = DataOut

    @classmethod
    def _get_tfidf(cls):
        return TfidfVectorizer(
            use_idf=True, norm='l2', analyzer='word', ngram_range=(1, 3)
        )

    def process_query(self, query):
        query = re.sub('\W+', ' ', query).strip()
        return self.pre_processor.process(query)

    def search(self, query, k):
        scores = list()
        tokens = self.process_query(query)
        query_vector = self._get_query_vector(tokens)
        for doc in self.matrix.A:
            scores.append(cosine_sim(query_vector, doc))

        return self._get_results(scores, k)

    def get_search_results_df(self, query, k):
        output = self.search(query, k)
        return self.output_cls.to_df(output)

    def _get_results(self, scores, k):
        out = np.array(scores).argsort()[-k:][::-1]
        return [self.output_cls(
            url=self.data[index]['url'],
            score=scores[index]
        ) for index in out]

    def _get_query_vector(self, tokens):
        n = len(self.vocabulary)
        vector = np.zeros(n)

        for token in itertools.chain(*tokens):
            try:
                index = self.tfidf.vocabulary_[token]
                vector[index] = 1
            except ValueError:
                pass
        return vector
