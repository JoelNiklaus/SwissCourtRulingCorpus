from __future__ import annotations
import configparser
from typing import Any, TYPE_CHECKING

from scrc.utils.abstract_extractor import AbstractExtractor
from root import ROOT_DIR
from scrc.utils.log_utils import get_logger

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame

class JudgementExtractor(AbstractExtractor):
    """
    Extracts the judgements from the rulings section
    """

    def __init__(self, config: dict):
        super().__init__(config, function_name='judgement_extracting_functions', col_name='judgements')
        self.logger = get_logger(__name__)
        self.processed_file_path = self.data_dir / "spiders_judgement_extracted.txt"
        self.logger_info = {
            'start': 'Started extracting judgements',
            'finished': 'Finished extracting judgements',
            'start_spider': 'Started extracting the judgements for spider',
            'finish_spider': 'Finished extracting the judgements for spider',
            'saving': 'Saving chunk of judgements',
            'processing_one': 'Extracting the judgement from',
            'no_functions': 'Not extracting the judgements.'
        }

    def get_database_selection_string(self, spider: str, lang: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        return f"spider='{spider}' AND rulings IS NOT NULL AND rulings <> ''"

    def get_required_data(self, series: DataFrame) -> Any:
        return series['rulings']

    def check_condition_before_process(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing.
        e.g. data is required to be present for analysis"""
        return bool(data)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    # this stops working when the script is called from the src directory!
    config.read(ROOT_DIR / 'config.ini')

    judgement_extractor = JudgementExtractor(config)
    judgement_extractor.start()
