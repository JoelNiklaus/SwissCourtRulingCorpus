from __future__ import annotations
from typing import Any, TYPE_CHECKING
import configparser

from root import ROOT_DIR
from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor
from scrc.utils.log_utils import get_logger

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame

class LowerCourtExtractor(AbstractExtractor):
    """
    Extracts the lower courts from the header section
    """

    def __init__(self, config: dict):
        super().__init__(config, function_name='lower_court_extracting_functions',
                         col_name='lower_court')
        self.logger = get_logger(__name__)
        self.processed_file_path = self.data_dir / "spiders_lower_court_extracted.txt"
        self.logger_info = {
            'start': 'Started extracting lower court informations',
            'finished': 'Finished extracting lower court informations',
            'start_spider': 'Started extracting lower court informations for spider',
            'finish_spider': 'Finished extracting lower court informations for spider',
            'saving': 'Saving chunk of lower court informations',
            'processing_one': 'Extracting lower court informations from',
            'no_functions': 'Not extracting lower court informations.'
        }

    def get_database_selection_string(self, spider: str, lang: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        return f"spider='{spider}' AND header IS NOT NULL AND header <> ''"

    def get_required_data(self, series: DataFrame) -> Any:
        return series['header']

    def check_condition_before_process(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing.
        e.g. data is required to be present for analysis"""
        return bool(data)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    # this stops working when the script is called from the src directory!
    config.read(ROOT_DIR / 'config.ini')

    lower_court_extractor = LowerCourtExtractor(config)
    lower_court_extractor.start()
