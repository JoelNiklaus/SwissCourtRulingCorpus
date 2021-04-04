import configparser

from root import ROOT_DIR
from scrc.dataset_construction.aggregator import Aggregator
from scrc.dataset_construction.cleaner import Cleaner
from scrc.dataset_construction.extractor import Extractor
from scrc.dataset_construction.kaggle_dataset_creator import KaggleDatasetCreator
from scrc.dataset_construction.mongo_db_importer import MongoDBImporter
from scrc.dataset_construction.scraper import Scraper, base_url


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
    aggregator.combine_spiders()

    kaggle_dataset_creator = KaggleDatasetCreator(config)
    kaggle_dataset_creator.create_dataset()

    mongo_db_importer = MongoDBImporter(config)
    mongo_db_importer.import_data()


if __name__ == '__main__':
    main()
