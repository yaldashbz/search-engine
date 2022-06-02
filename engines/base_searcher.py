from abc import ABC, abstractmethod

from engines.utils import DataOut
from pre_process import PreProcessor


class BaseSearcher(ABC):
    def __init__(self, data):
        self.data = data
        self.output_cls = DataOut
        self.pre_processor = PreProcessor()

    @abstractmethod
    def process_query(self, query):
        pass

    @abstractmethod
    def search(self, query, k):
        pass

    def get_search_results_df(self, query, k):
        output = self.search(query, k)
        return self.output_cls.to_df(output)
