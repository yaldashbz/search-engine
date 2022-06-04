import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm


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


class TransformerOut(DataOut):
    def __init__(self, url, distance):
        super().__init__(url, self._get_score(distance))

    @classmethod
    def _get_score(cls, distance):
        return 1 - distance / 2


def get_dict(big_list):
    return {word: i for i, word in enumerate(big_list)}


def get_all_urls_and_words(data):
    all_urls = list()
    contents = list()
    for d in data:
        all_urls.append(d['url'])
        contents.append(d['content'])
    all_words = list(set(' '.join(contents).split()))

    return get_dict(all_urls), get_dict(all_words)


def create_boolean_matrix(data, matrix_path: str, header_path: str):
    all_urls, all_words = get_all_urls_and_words(data)
    matrix = np.zeros(shape=(len(all_urls), len(all_words)), dtype=bool)
    header = {'rows': all_urls, 'columns': all_words}
    for d in tqdm(data):
        url: str = d['url']
        content: str = d['content']
        words = content.split()
        for word in words:
            matrix[all_urls[url]][all_words[word]] = True

    json.dump(header, open(header_path, 'w+'))
    np.savez_compressed(matrix_path, matrix=matrix)

    return matrix, header
