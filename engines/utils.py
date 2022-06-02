from dataclasses import dataclass

import numpy as np
import pandas as pd


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@dataclass
class DataOut:
    url: str
    score: float

    def __init__(self, url, score):
        self.url = url
        self.score = float(score)

    @classmethod
    def to_df(cls, output):
        return pd.DataFrame(output)


class TFIDFOut(DataOut):
    pass


class TransformerOut(DataOut):
    def __init__(self, url, distance):
        super().__init__(url, self._get_score(distance))

    @classmethod
    def _get_score(cls, distance):
        return 1 - distance / 2
