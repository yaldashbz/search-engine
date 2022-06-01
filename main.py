import json

from data.data import EngineData
from data.scrapers import WikiScraper


def run():
    data = WikiScraper().scrape(max_depth=2, use_linked=True)
    EngineData.save(data, 'z')
    print(data)


def check_size():
    data = json.load(open('z', 'r+'))
    print(len(data))


if __name__ == '__main__':
    # run()
    check_size()

