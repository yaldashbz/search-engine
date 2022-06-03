import os
import itertools
import numpy as np
from gensim.models.fasttext import FastText

from engines.base_searcher import BaseSearcher
from engines.utils import cosine_sim
from retriever.utils import get_contents, get_words


class FasttextSearcher(BaseSearcher):
    _EPOCHS = 6
    _MODEL_PATH = 'models'
    _MODEL_FILE = 'fasttext.model'

    def __init__(self, data, train: bool = True, min_count: int = 1):
        super().__init__(data)
        contents = get_contents(data)

        path = os.path.join(self._MODEL_PATH, self._MODEL_FILE)
        if not (train or os.path.exists(path)):
            raise ValueError

        self.fasttext = self._get_fasttext(train, min_count, path)
        if train:
            self._train(contents)
            self._save_model()
        self.doc_embedding_avg = self._get_doc_embedding_avg(contents)

    @classmethod
    def _get_fasttext(cls, train: bool, min_count: int, path: str):
        return FastText(
            sg=1, window=10, min_count=min_count,
            negative=15, min_n=2, max_n=5
        ) if train else FastText.load(path)

    def _train(self, contents):
        tokens = [get_words(content) for content in contents]
        self.fasttext.build_vocab(tokens)
        self.fasttext.train(
            tokens,
            epochs=self._EPOCHS,
            total_examples=self.fasttext.corpus_count,
            total_words=self.fasttext.corpus_total_words
        )

    def _save_model(self):
        if not os.path.exists(self._MODEL_PATH):
            os.mkdir(self._MODEL_PATH)
        self.fasttext.save(os.path.join(self._MODEL_PATH, self._MODEL_FILE))

    def _get_doc_embedding_avg(self, contents):
        docs_avg = dict()
        for index, content in enumerate(contents):
            words = get_words(content)
            docs_avg[index] = np.mean([self.fasttext.wv[word] for word in words])
        return docs_avg

    def _get_query_embedding_avg(self, tokens):
        return np.mean([self.fasttext.wv[token] for token in tokens])

    def process_query(self, query):
        tokens = self.pre_processor.process(query)
        return list(itertools.chain(*tokens))

    def search(self, query, k):
        tokens = self.process_query(query)
        query_embedding_avg = self._get_query_embedding_avg(tokens)
        similarities = dict()
        for index, embedding in self.doc_embedding_avg.items():
            similarities[index] = cosine_sim(embedding, query_embedding_avg)

        similarities = sorted(similarities.items(), key=lambda x: x[1])[::-1][:k]
        return self._get_result(similarities)

    def _get_result(self, similarities):
        return [self.output_cls(
            url=self.data[index]['url'],
            score=score
        ) for index, score in similarities]
