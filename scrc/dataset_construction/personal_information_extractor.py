import configparser
from typing import Optional, Any

from root import ROOT_DIR
from scrc.utils.abstract_extractor import AbstractExtractor
import pandas as pd

class PersonalInformationExtractor(AbstractExtractor):
    """
    Extracts the lower courts from the header section
    """

    def __init__(self, config: dict):
        super().__init__(config, function_name='personal_information_extracting_functions', col_name='personal_informations')
        self.languages = ['it']

    def getDatabaseSelectionString(self, spider, lang) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        return f"spider='{spider}' AND header IS NOT NULL AND header <> ''"

    def getRequiredData(self, series) -> Any:
        return series['header']

    def checkConditionBeforeProcess(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing. 
        e.g. data is required to be present for analysis"""
        return bool(data)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    personal_information_extractor = PersonalInformationExtractor(config)
    personal_information_extractor.processed_file_path = personal_information_extractor.data_dir / "spiders_personal_information_extracted.txt"
    personal_information_extractor.loggerInfo = {
        'start': 'Started extracting personal informations', 
        'finished': 'Finished extracting personal informations', 
        'start_spider': 'Started extracting personal informations for spider', 
        'finish_spider': 'Finished extracting personal informations for spider', 
        'saving': 'Saving chunk of personal informations',
        'processing_one': 'Extracting personal informations from',
        'no_functions': 'Not extracting personal informations.'
        }
    personal_information_extractor.start()
