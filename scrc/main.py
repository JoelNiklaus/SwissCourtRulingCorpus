import configparser

from root import ROOT_DIR
from scrc.aggregator import Aggregator
from scrc.cleaner import Cleaner
from scrc.extractor import Extractor
from scrc.scraper import Scraper, base_url


def main():
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    scraper = Scraper(config)
    scraper.download_subfolders(base_url + "docs/")

    extractor = Extractor(config)
    extractor.build_dataset()

    cleaner = Cleaner(config)
    cleaner.clean()

    aggregator = Aggregator(config)
    aggregator.combine_courts()


if __name__ == '__main__':
    main()
