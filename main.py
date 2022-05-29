from data.data import EngineData
from data.scrapers import WikiScraper


def run():
    data = WikiScraper().scrape()
    EngineData.save(data, 'z')
    print(data)


if __name__ == '__main__':
    run()
