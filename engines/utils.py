import json
from dataclasses import dataclass
from itertools import chain

import numpy as np
import pandas as pd
from tqdm import tqdm

from retriever.utils import get_words


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@dataclass
class DataOut:
    url: str
    score: float

    def __init__(self, url, score):
        self.url = url
        self.score = float(score)

    @classmethod
    def to_df(cls, output):
        return pd.DataFrame(output)


class BooleanDataOut(DataOut):
    def __init__(self, url):
        super().__init__(url, score=1)


def get_dict(big_list):
    return {word: i for i, word in enumerate(big_list)}


def get_all_urls_and_words(data):
    all_urls = list()
    all_words = list()
    for doc in data:
        all_urls.append(doc['url'])
        all_words += get_words(doc)
    all_urls = list(set(all_urls))
    all_words = list(set(all_words))
    return get_dict(all_urls), get_dict(all_words)


def create_boolean_matrix(data, matrix_path: str, header_path: str):
    all_urls, all_words = get_all_urls_and_words(data)
    matrix = np.zeros(shape=(len(all_urls), len(all_words)), dtype=bool)
    header = {'rows': all_urls, 'columns': all_words}
    for doc in tqdm(data):
        url = doc['url']
        words = get_words(doc)
        for word in words:
            matrix[all_urls[url]][all_words[word]] = True

    json.dump(header, open(header_path, 'w+'))
    np.savez_compressed(matrix_path, matrix=matrix)

    return matrix, header
