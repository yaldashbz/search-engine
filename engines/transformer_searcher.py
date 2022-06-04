import itertools

import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from engines.base_searcher import BaseSearcher
from engines.utils import cosine_sim
from retriever.utils import get_sentences


class TransformerSearcher(BaseSearcher):
    def __init__(self, data):
        super().__init__(data)

        self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        self._to_cuda()
        self.embeddings = self._get_embeddings()

    def _to_cuda(self):
        if torch.cuda.is_available():
            self.model = self.model.to(torch.device('cuda'))

    def _get_embeddings(self):
        embeddings = dict()
        for doc in tqdm(self.data):
            sentences = get_sentences(doc)
            embedding = self.model.encode(sentences, normalize_embeddings=True)
            embeddings[doc['url']] = embedding
        return embeddings

    def process_query(self, query):
        tokens = self.pre_processor.process(query)
        tokens = itertools.chain(*tokens)
        query = ' '.join(tokens)
        return [query]

    def search(self, query, k):
        query = self.process_query(query)
        query_embedding = self.model.encode(query, show_progress_bar=True, normalize_embeddings=True)

        similarities = dict()
        for url, embedding in self.embeddings.items():
            similarities[url] = cosine_sim(np.mean(embedding, axis=0).reshape(1, 768), query_embedding.reshape(768, 1))

        similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
        return self._get_result(similarities)

    def _get_result(self, similarities):
        return [self.output_cls(
            url=url,
            score=score
        ) for url, score in similarities]
