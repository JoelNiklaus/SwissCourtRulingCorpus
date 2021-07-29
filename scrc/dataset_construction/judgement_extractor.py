import configparser
from scrc.utils.abstract_extractor import Extractor
from typing import Optional, Any

from root import ROOT_DIR
import pandas as pd


class JudgementExtractor(Extractor):
    """
    Extracts the judgements from the rulings section
    """

    def __init__(self, config: dict):
        super().__init__(config, function_name='judgement_extracting_functions', col_name='judgements')

    def getDatabaseSelectionString(self, spider, lang) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        return f"spider='{spider}' AND rulings IS NOT NULL AND rulings <> ''"

    def getRequiredData(self, series) -> Any:
        return series['rulings']

    def checkConditionBeforeProcess(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing. 
        e.g. data is required to be present for analysis"""
        return bool(data)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    judgement_extractor = JudgementExtractor(config)
    judgement_extractor.processed_file_path = judgement_extractor.data_dir / "spiders_judgement_extracted.txt"
    judgement_extractor.loggerInfo = {
        'start': 'Started extracting judgements', 
        'finished': 'Finished extracting judgements', 
        'start_spider': 'Started extracting the judgements for spider', 
        'finish_spider': 'Finished extracting the judgements for spider', 
        'saving': 'Saving chunk of judgements',
        'processing_one': 'Extracting the judgement from',
        'no_functions': 'Not extracting the judgements.'
        }
    judgement_extractor.start()
