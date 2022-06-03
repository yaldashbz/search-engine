import json
from dataclasses import dataclass
from typing import List

from pre_process import PreProcessor
from retriever.utils import get_keywords, DIVIDER


@dataclass
class EngineData:
    url: str
    content: str
    keywords: List[str]

    def __init__(self, url, content):
        self.url = url
        self.content = DIVIDER.join([DIVIDER.join(sentence) for sentence in PreProcessor().process(content)])
        self.keywords = get_keywords(self.content)

    def __hash__(self):
        return hash(f'{self.url} - {self.content}')

    @classmethod
    def _convert(cls, data: List) -> List:
        return [page.__dict__ for page in data]

    @classmethod
    def _cleanup(cls, data: List) -> List:
        return [doc for doc in data if doc['content'] != '']

    @classmethod
    def save(cls, data: List, path: str):
        data = cls._cleanup(data)
        json.dump(cls._convert(data), open(path, 'a+'))
