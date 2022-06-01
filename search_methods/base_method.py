from abc import ABC, abstractmethod

from pre_process import PreProcessor


class BaseSearcher(ABC):
    def __init__(self):
        self.pre_processor = PreProcessor()

    @abstractmethod
    def process_query(self, query):
        pass

    @abstractmethod
    def search(self):
        pass
