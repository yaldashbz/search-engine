from typing import List

import requests
from bs4 import BeautifulSoup

from data.data import EngineData


class BaseWebScraper:

    def scrape(self, url: str) -> List[EngineData]:
        """Get data from a web page: url"""
        raise NotImplementedError

    @classmethod
    def get_soup(cls, url: str) -> BeautifulSoup:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
