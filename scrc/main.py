import configparser
import faulthandler

from root import ROOT_DIR
from scrc.dataset_construction.citation_extractor import CitationExtractor
from scrc.dataset_construction.cleaner import Cleaner
from scrc.dataset_construction.extractor import Extractor
from scrc.dataset_construction.kaggle_dataset_creator import KaggleDatasetCreator
from scrc.dataset_construction.scraper import Scraper, base_url
from scrc.dataset_construction.section_splitter import SectionSplitter
from scrc.dataset_construction.spacy_pipeline_runner import SpacyPipelineRunner
from scrc.dataset_construction.count_computer import CountComputer

from filprofiler.api import profile


"""
New approach:
- Scrape data into spider folders
- Extract text and metadata content (save to Postgres DB because of easy interoperability with pandas (filtering and streaming))
- Clean text  (keep raw content in db)
- Process each text with spacy, save doc to disk and store path in db, store num token count in separate db col
doc.to_disk("/path/to/doc", exclude=['tensor']) # think about excluding tensor to save space (almost 4x less space)
- compute lemma counts and save aggregates in separate tables
- split BGer into sections (from html_raw)
- extract BGer citations (from html_raw) using "artref" tags

TODO
- extract judgement 


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

    count_computer = CountComputer(config)
    count_computer.run_pipeline()

    section_splitter = SectionSplitter(config)
    section_splitter.split_sections()

    citation_extractor = CitationExtractor(config)
    citation_extractor.extract_citations()

    # kaggle_dataset_creator = KaggleDatasetCreator(config)
    # kaggle_dataset_creator.create_dataset()


if __name__ == '__main__':
    main()
