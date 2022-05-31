from abc import ABC, abstractmethod
from typing import List

import requests
from bs4 import BeautifulSoup

from retriever.data import EngineData


class BaseWebScraper(ABC):

    @abstractmethod
    def scrape(self, url: str) -> List[EngineData]:
        """Get data from a web page: url"""
        raise NotImplementedError

    @classmethod
    def get_soup(cls, url: str) -> BeautifulSoup:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
