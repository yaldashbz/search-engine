from collections import defaultdict
from typing import List

from pre_process import PreProcessor

WIKI_CATEGORIES = [
    'bbc', 'religion', 'sport', 'drink',
    'financial', 'health', 'literature',
    'social networks', 'food', 'history',
    'animal'
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
