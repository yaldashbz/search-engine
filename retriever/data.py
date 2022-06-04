import json
from dataclasses import dataclass
from typing import List

from pre_process import PreProcessor
from retriever.utils import get_keywords, DIVIDER


@dataclass
class EngineData:
    url: str
    tokens: List[List[str]]
    keywords: List[str]
    content: str

    def __init__(self, url, content):
        self.url = url
        self.tokens = PreProcessor().process(content)
        self.content = DIVIDER.join([DIVIDER.join(sentence) for sentence in self.tokens])
        self.keywords = get_keywords(content)

    def __hash__(self):
        return hash(f'{self.url} - {self.content}')

    @classmethod
    def _convert(cls, data: List) -> List:
        return [{'url': doc.url, 'tokens': doc.tokens} for doc in data]

    @classmethod
    def _cleanup(cls, data: List) -> List:
        return [doc for doc in data if doc.tokens]

    @classmethod
    def save(cls, data: List, path: str):
        data = cls._cleanup(data)
        json.dump(cls._convert(data), open(path, 'w+'))
