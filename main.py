import json
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from data.data import EngineData
from data.scrapers import WikiScraper


def run():
    data = WikiScraper().scrape(max_depth=2, use_linked=True)
    EngineData.save(data, 'data_files/z.json')


def check_size():
    data = json.load(open('data_files/z.json', 'r+'))
    print(len(data))


def get_all_urls_and_words(data):
    all_urls = list()
    contents = list()
    for d in data:
        all_urls.append(d['url'])
        contents.append(d['content'])
    all_words = list(set(' '.join(contents).split()))

    return all_urls, all_words


def create_boolean_matrix():
    data = json.load(open('data_files/z.json', 'r+'))
    all_urls, all_words = get_all_urls_and_words(data)
    matrix = np.zeros(shape=(len(all_urls), len(all_words)), dtype=bool)
    header = {'rows': all_urls, 'columns': all_words}
    for d in tqdm(data):
        url: str = d['url']
        content: str = d['content']
        words = content.split()
        for word in words:
            matrix[all_urls.index(url)][all_words.index(word)] = True

    json.dump(header, open('data_files/boolean_matrix/header.json', 'w+'))
    np.savez_compressed('data_files/boolean_matrix/test', matrix=matrix)

    return matrix


def load_boolean_matrix():
    header: Dict[str, List] = json.load(open('data_files/boolean_matrix/header.json', 'r+'))
    matrix = np.load('data_files/boolean_matrix/test.npz')['matrix']
    print([i for i in range(len(header['rows'])) if matrix[i][header['columns'].index('freedom')]])


if __name__ == '__main__':
    run()
    # check_size()
    # create_boolean_matrix()
    # load_boolean_matrix()
