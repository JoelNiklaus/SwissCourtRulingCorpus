import configparser
import faulthandler

from root import ROOT_DIR
from scrc.dataset_construction.cleaner import Cleaner
from scrc.dataset_construction.extractor import Extractor
from scrc.dataset_construction.kaggle_dataset_creator import KaggleDatasetCreator
from scrc.dataset_construction.scraper import Scraper, base_url
from scrc.dataset_construction.spacy_pipeline_runner import SpacyPipelineRunner
from scrc.dataset_construction.vocabulary_computer import VocabularyComputer

from filprofiler.api import profile


"""
New approach:
- Scrape data into spider folders
- Extract text and metadata content (save to Postgres DB because of easy interoperability with pandas (filtering and streaming))
- Clean text  (keep raw content in db)
- Process each text with spacy, save doc to disk and store path in db, store num token count in separate db col
doc.to_disk("/path/to/doc", exclude=['tensor']) # think about excluding tensor to save space (almost 4x less space)
- split BGer into sections (from html_raw)
- extract citations (from raw text)


"""


def main():
    # faulthandler.enable()  # can print a minimal threaddump in case of external termination

    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    scraper = Scraper(config)
    scraper.download_subfolders(base_url + "docs/")

    extractor = Extractor(config)
    extractor.build_dataset()

    cleaner = Cleaner(config)
    cleaner.clean()

    spacy_pipeline_runner = SpacyPipelineRunner(config)
    spacy_pipeline_runner.run_pipeline()

    vocabulary_computer = VocabularyComputer(config)
    vocabulary_computer.run_pipeline()

    # kaggle_dataset_creator = KaggleDatasetCreator(config)
    # kaggle_dataset_creator.create_dataset()


if __name__ == '__main__':
    main()
