from typing import List, Tuple, Set

import wikipedia
from tqdm import tqdm
from wikipedia import WikipediaException

from retriever.data import EngineData
from retriever.scrapers.base_scraper import BaseWebScraper
from retriever.utils import WIKI_CATEGORIES

wikipedia.set_lang('en')
wikipedia.set_rate_limiting(True)


class WikiScraper(BaseWebScraper):
    def __init__(self):
        self.titles = []  # TODO: remove
        self.categories = WIKI_CATEGORIES

    def scrape(
            self,
            url: str = None,
            max_depth: int = 3,
            use_linked: bool = False
    ) -> List[EngineData]:

        return self._linked_scrape(max_depth) if use_linked else self._scrape(max_depth)

    def _scrape(self, max_depth) -> List[EngineData]:
        self.titles = self._get_titles(list(), self.categories, max_depth)
        return self._get_data(self.titles)

    def _linked_scrape(self, max_depth) -> List[EngineData]:
        titles = self._get_titles(list(), self.categories, max_depth)
        linked_titles, base_data = self._get_linked_titles(titles)
        self.titles = titles.union(linked_titles)
        linked_data = self._get_data(linked_titles)
        return list(set(base_data).union(linked_data))

    @classmethod
    def _get_titles(cls, titles, subjects, max_depth) -> Set[str]:
        if max_depth == 0:
            return set(titles)

        for subject in tqdm(subjects):
            titles += wikipedia.search(subject)

        new_subjects = titles.copy()
        return cls._get_titles(titles, new_subjects, max_depth - 1)

    @classmethod
    def _get_linked_titles(cls, titles) -> Tuple[Set[str], List[EngineData]]:
        links = []
        base_data = []
        for title in titles:
            try:
                page = wikipedia.page(title)
                links += page.links
                base_data.append(EngineData(url=page.url, content=page.content))
            except WikipediaException:
                continue
        return set(links), base_data

    @classmethod
    def _get_data(cls, titles) -> List[EngineData]:
        data = []
        for title in tqdm(titles):
            try:
                page = wikipedia.page(title)
                data.append(EngineData(url=page.url, content=page.content))
            except WikipediaException:
                continue
        return data
