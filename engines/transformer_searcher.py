import itertools

import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from engines.base_searcher import BaseSearcher
from engines.utils import TransformerOut
from retriever.utils import get_contents


class TransformerSearcher(BaseSearcher):
    def __init__(self, data):
        super().__init__(data)
        contents = get_contents(data)

        self.output_cls = TransformerOut
        self.data_len = len(contents)
        self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        self._to_cuda()
        embeddings = self._get_embeddings(contents)
        self.index = self._get_index(embeddings)

    def _to_cuda(self):
        if torch.cuda.is_available():
            self.model = self.model.to(torch.device('cuda'))

    def _get_embeddings(self, contents):
        embeddings = self.model.encode(contents, show_progress_bar=True, normalize_embeddings=True)
        return np.array([embedding for embedding in embeddings]).astype('float32')

    def _get_index(self, embeddings):
        index = faiss.IndexIDMap(faiss.IndexFlatL2(embeddings.shape[1]))
        index.add_with_ids(embeddings, np.array(range(self.data_len)))
        return index

    def process_query(self, query):
        tokens = self.pre_processor.process(query)
        tokens = itertools.chain(*tokens)
        query = ' '.join(tokens)
        return [query]

    def search(self, query, k):
        query = self.process_query(query)
        vector = self.model.encode(query, show_progress_bar=True, normalize_embeddings=True)
        distances, indexes = self.index.search(np.array(vector).astype('float32'), k=k)
        return self._get_results(distances, indexes)

    def _get_results(self, distances, indexes):
        indexes = indexes.flatten().tolist()
        distances = distances.flatten().tolist()
        return [self.output_cls(
            url=self.data[index]['url'],
            distance=distances[i]
        ) for i, index in enumerate(indexes)]
