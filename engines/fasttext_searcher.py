import os
import itertools
import numpy as np
from gensim.models.fasttext import FastText

from engines.base_searcher import BaseSearcher
from engines.utils import cosine_sim, DataOut
from retriever.utils import get_contents, get_words


class FasttextSearcher(BaseSearcher):
    _EPOCHS = 6
    _MODEL_PATH = 'models/fasttext.model'

    def __init__(self, data, train=True):
        super().__init__(data)
        contents = get_contents(data)

        self.output_cls = DataOut
        self.fasttext = self._get_fasttext(train)
        if train:
            self._train(contents)

        self.doc_embedding_avg = self._get_doc_embedding_avg(contents)

    @classmethod
    def _get_fasttext(cls, train):
        return FastText(
            sg=1, window=10, min_count=1,
            negative=15, min_n=2, max_n=5
        ) if train else FastText.load(cls._MODEL_PATH)

    def _train(self, contents):
        tokens = [get_words(content) for content in contents]
        self.fasttext.build_vocab(tokens)
        self.fasttext.train(
            tokens,
            epochs=self._EPOCHS,
            total_examples=self.fasttext.corpus_count,
            total_words=self.fasttext.corpus_total_words
        )
        if not os.path.exists(self._MODEL_PATH):
            os.mkdir(self._MODEL_PATH)
        self.fasttext.save(self._MODEL_PATH)

    def _get_doc_embedding_avg(self, contents):
        docs_avg = dict()
        for i, content in enumerate(contents):
            words = get_words(content)
            total = np.zeros(len(words))
            for word in words:
                total = np.sum([total, self.fasttext[word]], axis=0)
            docs_avg[i] = total / len(words)
        return docs_avg

    def _get_query_embedding_avg(self, tokens):
        total = np.zeros(len(tokens))
        for token in tokens:
            total = np.sum([total, self.fasttext[token]], axis=0)
        return total / len(tokens)

    def process_query(self, query):
        tokens = self.pre_processor.process(query)
        return list(itertools.chain(*tokens))

    def search(self, query, k):
        tokens = self.process_query(query)
        query_embedding_avg = self._get_query_embedding_avg(tokens)
        similarities = dict()
        for index, embedding in self.doc_embedding_avg.items():
            similarities[index] = cosine_sim(embedding, query_embedding_avg)

        similarities = sorted(similarities, key=lambda x: x[1])[::-1]
        return self._get_result(similarities)

    def get_search_results_df(self, query, k):
        output = self.search(query, k)
        return self.output_cls.to_df(output)

    def _get_result(self, similarities):
        return [self.output_cls(
            url=self.data[index]['url'],
            score=score
        ) for index, score in similarities]
