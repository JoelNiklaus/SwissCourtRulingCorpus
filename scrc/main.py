import configparser
import faulthandler

from root import ROOT_DIR
from scrc.dataset_construction.citation_extractor import CitationExtractor
from scrc.dataset_construction.cleaner import Cleaner
from scrc.dataset_construction.extractor import Extractor
from scrc.dataset_construction.judgement_extractor import JudgementExtractor
from scrc.dataset_construction.dataset_creator import DatasetCreator
from scrc.dataset_construction.scraper import Scraper, base_url
from scrc.dataset_construction.section_splitter import SectionSplitter
from scrc.dataset_construction.spacy_pipeline_runner import SpacyPipelineRunner
from scrc.dataset_construction.count_computer import CountComputer

from filprofiler.api import profile

from scrc.external_corpora.jureko_processor import JurekoProcessor
from scrc.external_corpora.slc_processor import SlcProcessor
from scrc.external_corpora.wikipedia_processor import WikipediaProcessor
from scrc.utils.decorators import slack_alert

"""
This file aggregates all the pipeline components and can be a starting point for running the entire pipeline.

Approach:
- Scrape data into spider folders
- Extract text and metadata content (save to Postgres DB because of easy interoperability with pandas (filtering and streaming))
- Clean text  (keep raw content in db)
- Split BGer into sections (from html_raw)
- Extract BGer citations (from html_raw) using "artref" tags
- Extract judgements 
- Process each text with spacy, save doc to disk and store path in db, store num token count in separate db col
- Compute lemma counts and save aggregates in separate tables
- Create the smaller datasets derived from SCRC with the available metadata
"""


@slack_alert
def main():
    """
    Runs the entire pipeline.
    :return:
    """
    # faulthandler.enable()  # can print a minimal threaddump in case of external termination

    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    process_scrc(config)

    process_external_corpora(config)


def process_scrc(config):
    """
    Processes everything related to the core SCRC corpus
    :param config:
    :return:
    """
    scraper = Scraper(config)
    scraper.download_subfolders(base_url + "docs/")

    extractor = Extractor(config)
    extractor.build_dataset()

    cleaner = Cleaner(config)
    cleaner.clean()

    section_splitter = SectionSplitter(config)
    section_splitter.split_sections()

    citation_extractor = CitationExtractor(config)
    citation_extractor.extract_citations()

    judgement_extractor = JudgementExtractor(config)
    judgement_extractor.extract_judgements()

    spacy_pipeline_runner = SpacyPipelineRunner(config)
    spacy_pipeline_runner.run_pipeline()

    count_computer = CountComputer(config)
    count_computer.run_pipeline()

    dataset_creator = DatasetCreator(config)
    dataset_creator.create_datasets()


def process_external_corpora(config):
    """
    Processes external corpora which can be compared to SCRC.
    :param config:
    :return:
    """
    wikipedia_processor = WikipediaProcessor(config)
    wikipedia_processor.process()

    jureko_processor = JurekoProcessor(config)
    jureko_processor.process()

    slc_processor = SlcProcessor(config)
    slc_processor.process()


if __name__ == '__main__':
    main()
