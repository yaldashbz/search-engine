from collections import defaultdict
from itertools import chain
from typing import List

from pre_process import PreProcessor

DIVIDER = ' '

CATEGORIES = [
    'religion',
    'sports', 'drink',
    'financial', 'health', 'literature',
    'social networks', 'food', 'history',
    'animals', 'news', 'science', 'movies',
    'music', 'games', 'computer',
    'football', 'basketball', 'volleyball',
    'university', 'national', 'politics'
]


def get_keywords(content: str) -> List[str]:
    words = list()
    for sentence in PreProcessor.tokenize(content):
        words += sentence
    count = defaultdict(int)
    for word in words:
        word = word.replace(',', '').replace('.', '').replace('?', '').replace('!', '').replace(';', '')
        count[word] += 1
    return list({k: v for k, v in sorted(count.items(), key=lambda item: item[1])}.keys())[-20:]


def get_contents(data: List):
    return [
        DIVIDER.join([DIVIDER.join(sentence) for sentence in doc])
        for doc in data
    ]


def get_sentences(doc):
    return [' '.join(words) for words in doc['tokens']]


def get_words(doc):
    return list(chain(*doc['tokens']))
