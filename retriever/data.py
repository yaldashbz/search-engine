import json
from dataclasses import dataclass
from typing import List

from retriever.utils import get_keywords
from pre_process import PreProcessor


@dataclass
class EngineData:
    url: str
    sentences: str
    keywords: List[str]

    def __init__(self, url, content):
        self.url = url
        self.data = PreProcessor().process(content)
        content = ' '.join([' '.join(sentence) for sentence in self.data])
        self.keywords = get_keywords(content)

    @classmethod
    def _convert(cls, data: List) -> List:
        return [page.__dict__ for page in data]

    @classmethod
    def save(cls, data: List, path: str):
        json.dump(cls._convert(data), open(path, 'w+'))
