import json
from dataclasses import dataclass
from typing import List

from retriever.utils import get_keywords


@dataclass
class EngineData:
    url: str
    content: str
    keywords: List[str]

    def __init__(self, url, sentences):
        self.url = url
        self.content = ' '.join([' '.join(sentence) for sentence in sentences])
        self.keywords = get_keywords(self.content)

    @classmethod
    def _convert(cls, data: List) -> List:
        return [page.__dict__ for page in data]

    @classmethod
    def save(cls, data: List, path: str):
        json.dump(cls._convert(data), open(path, 'w+'))
