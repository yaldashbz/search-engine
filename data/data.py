import pandas as pd

from dataclasses import dataclass
from typing import List


@dataclass
class EngineData:
    url: str
    content: str

    @classmethod
    def _convert(cls, data: List) -> List:
        return [page.__dict__ for page in data]

    @classmethod
    def save(cls, data: List, path: str):
        df = pd.DataFrame(cls._convert(data))
        df.to_csv(path)
        return df
