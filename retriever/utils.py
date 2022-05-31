import itertools
from collections import defaultdict
from typing import List

WIKI_CATEGORIES = [
    'bbc', 'religion', 'sport',
    'financial', 'health', 'literature',
    'social networks'
]


def get_all_sentences(content: str):
    return list(itertools.chain(content))


def get_all_words(content: str):
    sentences = get_all_sentences(content)
    return list(itertools.chain(*sentences))


def get_keywords(content: str) -> List[str]:
    words = get_all_words(content)
    count = defaultdict(int)
    for word in words:
        count[word] += 1
    return list({k: v for k, v in sorted(count.items(), key=lambda item: item[1])}.keys())[-20:]
