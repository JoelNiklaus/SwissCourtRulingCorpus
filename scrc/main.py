
from typing import List, Dict
from scrc.preprocessors.language_identifier import LanguageIdentifier
from scrc.preprocessors.name_to_gender import NameToGender
from scrc.preprocessors.extractors.procedural_participation_extractor import ProceduralParticipationExtractor
from scrc.preprocessors.extractors.court_composition_extractor import CourtCompositionExtractor

from scrc.preprocessors.extractors.citation_extractor import CitationExtractor
from scrc.preprocessors.extractors.cleaner import Cleaner
from scrc.dataset_creation.doc2doc_ir_dataset_creator import Doc2DocIRDatasetCreator
from scrc.dataset_creation.criticality_dataset_creator import CriticalityDatasetCreator
from scrc.dataset_creation.judgment_dataset_creator import JudgmentDatasetCreator
from scrc.preprocessors.text_to_database import TextToDatabase
from scrc.preprocessors.extractors.judgment_extractor import JudgmentExtractor
from scrc.preprocessors.extractors.lower_court_extractor import LowerCourtExtractor
from scrc.preprocessors.scraper import Scraper, base_url
from scrc.preprocessors.extractors.section_splitter import SectionSplitter
from scrc.preprocessors.nlp_pipeline_runner import NlpPipelineRunner
from scrc.preprocessors.count_computer import CountComputer

from scrc.preprocessors.external_corpora.jureko_processor import JurekoProcessor
from scrc.preprocessors.external_corpora.slc_processor import SlcProcessor
from scrc.preprocessors.external_corpora.wikipedia_processor import WikipediaProcessor
from scrc.utils.decorators import slack_alert
from scrc.utils.main_utils import get_config

"""
This file aggregates all the pipeline components and can be a starting point for running the entire pipeline.

Approach:
- Scrape data into spider folders
- Extract text and metadata content (save to Postgres DB because of easy interoperability with pandas (filtering and streaming))
- Clean text  (keep raw content in db)
- Split BGer into sections (from html_raw)
- Extract BGer citations (from html_raw) using "artref" tags
- Extract judgments
- Process each text with spacy, save doc to disk and store path in db, store num token count in separate db col
- Compute lemma counts and save aggregates in separate tables
- Create the smaller datasets derived from SCRC with the available metadata
"""


def main():
    """
    Runs the entire pipeline.
    :return:
    """
    # faulthandler.enable()  # can print a minimal threaddump in case of external termination

    config = get_config()
    process_scrc(config)

    process_external_corpora(config)


def process_scrc(config):
    """
    Processes everything related to the core SCRC corpus
    :param config:
    :return:
    """
    construct_base_dataset(config)

    create_specialized_datasets(config)


def create_specialized_datasets(config):
    # TODO for identifying decisions in the datasets it would make sense to assign them a uuid.
    #  This could be based on other properties. We just need to make sure that we don't have any duplicates

    judgment_dataset_creator = JudgmentDatasetCreator(config)
    judgment_dataset_creator.create_dataset()

    citation_dataset_creator = Doc2DocIRDatasetCreator(config)
    citation_dataset_creator.create_dataset()

    criticality_dataset_creator = CriticalityDatasetCreator(config)
    criticality_dataset_creator.create_dataset()


def construct_base_dataset(config):   
    
    
    scraper = Scraper(config)
    new_files = scraper.download_subfolders(base_url + "docs/")
    
    text_to_database = TextToDatabase(config, new_files_only=True)
    text_to_database.build_dataset()
    
   
    language_identifier = LanguageIdentifier(config)
    decision_ids = language_identifier.start() 
    
    cleaner = Cleaner(config)
    cleaner.clean(decision_ids)
 
    
    section_splitter = SectionSplitter(config)
    section_splitter.start(decision_ids)

    citation_extractor = CitationExtractor(config)
    citation_extractor.start(decision_ids)

    judgment_extractor = JudgmentExtractor(config)
    judgment_extractor.start(decision_ids)

    lower_court_extractor = LowerCourtExtractor(config)
    lower_court_extractor.start(decision_ids)

    court_composition_extractor = CourtCompositionExtractor(config)
    court_composition_extractor.start(decision_ids)

    procedural_participation_extractor = ProceduralParticipationExtractor(config)
    procedural_participation_extractor.start(decision_ids)

    #name_to_gender = NameToGender(config)
    #name_to_gender.start()

    nlp_pipeline_runner = NlpPipelineRunner(config)
    nlp_pipeline_runner.run_pipeline()

    count_computer = CountComputer(config)
    count_computer.run_pipeline()


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
