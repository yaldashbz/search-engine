import os
import itertools
import numpy as np
from gensim.models.fasttext import FastText

from engines.base_searcher import BaseSearcher
from engines.utils import cosine_sim, DataOut
from retriever.utils import get_contents, get_words


class FasttextSearcher(BaseSearcher):
    _EPOCHS = 6
    _MODEL_PATH = '/models'
    _MODEL_FILE = 'fasttext.model'

    def __init__(self, data, train: bool = True, min_count: int = 1):
        super().__init__(data)
        contents = get_contents(data)

        self.output_cls = DataOut
        if train:
            self._train(contents, min_count)
        self.model = FastText.load(os.path.join(self._MODEL_PATH, self._MODEL_FILE))
        self.doc_embedding_avg = self._get_doc_embedding_avg(contents)

    def _train(self, contents, min_count):
        fasttext = FastText(
            sg=1, window=10, min_count=min_count,
            negative=15, min_n=2, max_n=5
        )
        tokens = [get_words(content) for content in contents]
        fasttext.build_vocab(tokens)
        fasttext.train(
            tokens,
            epochs=self._EPOCHS,
            total_examples=fasttext.corpus_count,
            total_words=fasttext.corpus_total_words
        )
        if not os.path.exists(self._MODEL_PATH):
            os.mkdir(self._MODEL_PATH)
        fasttext.save(os.path.join(self._MODEL_PATH, self._MODEL_FILE))

    def _get_doc_embedding_avg(self, contents):
        docs_avg = dict()
        for i, content in enumerate(contents):
            words = get_words(content)
            total = np.zeros(100)
            for word in words:
                total = np.sum([total, self.model.wv[word]], axis=0)
            docs_avg[i] = total / len(words)
        return docs_avg

    def _get_query_embedding_avg(self, tokens):
        total = np.zeros(100)
        for token in tokens:
            total = np.sum([total, self.model.wv[token]], axis=0)
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
