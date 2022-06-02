from dataclasses import dataclass

import numpy as np
import pandas as pd


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@dataclass
class TFIDFOut:
    url: str
    score: float

    def __init__(self, url, score):
        self.url = url
        self.score = float(score)

    @classmethod
    def to_df(cls, output):
        return pd.DataFrame(output)
