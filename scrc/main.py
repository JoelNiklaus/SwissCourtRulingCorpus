import configparser

from root import ROOT_DIR
from scrc.dataset_construction.aggregator import Aggregator
from scrc.dataset_construction.cleaner import Cleaner
from scrc.dataset_construction.extractor import Extractor
from scrc.dataset_construction.kaggle_dataset_creator import KaggleDatasetCreator
from scrc.dataset_construction.mongo_db_importer import MongoDBImporter
from scrc.dataset_construction.scraper import Scraper, base_url
from scrc.dataset_construction.spacy_pipeline_runner import SpacyPipelineRunner
from scrc.dataset_construction.splitter import Splitter


def main():
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    scraper = Scraper(config)
    scraper.download_subfolders(base_url + "docs/")

    extractor = Extractor(config)
    extractor.build_dataset()

    cleaner = Cleaner(config)
    cleaner.clean()

    splitter = Splitter(config)
    splitter.split_spiders()

    aggregator = Aggregator(config)
    aggregator.combine_spiders()

    mongo_db_importer = MongoDBImporter(config)
    mongo_db_importer.import_data()

    spacy_pipeline_runner = SpacyPipelineRunner(config)
    spacy_pipeline_runner.run_pipeline()

    kaggle_dataset_creator = KaggleDatasetCreator(config)
    kaggle_dataset_creator.create_dataset()


if __name__ == '__main__':
    main()
