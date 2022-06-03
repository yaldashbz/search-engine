from collections import defaultdict
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


def get_words(content: str):
    return content.split(DIVIDER)


def get_contents(data: List):
    return [doc['content'] for doc in data]
