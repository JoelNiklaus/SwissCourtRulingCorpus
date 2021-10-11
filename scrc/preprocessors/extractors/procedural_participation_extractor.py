from __future__ import annotations
import configparser
from typing import Any, TYPE_CHECKING

from root import ROOT_DIR
from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame

class ProceduralParticipationExtractor(AbstractExtractor):
    """
    Extracts the parties from the header section. This is part of the judicial person extraction task.
    """

    def __init__(self, config: dict):
        super().__init__(config, function_name='procedural_participation_extracting_functions', col_name='parties')
        self.processed_file_path = self.data_dir / "spiders_procedural_participation_extracted.txt"
        self.logger_info = {
        'start': 'Started extracting the involved parties',
        'finished': 'Finished extracting the involved parties',
        'start_spider': 'Started extracting the involved parties for spider',
        'finish_spider': 'Finished extracting the involved parties for spider',
        'saving': 'Saving chunk of involved parties',
        'processing_one': 'Extracting the involved parties from',
        'no_functions': 'Not extracting the procedural participation.'
        }

    def get_database_selection_string(self, spider: str, lang: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        return f"spider='{spider}' AND header IS NOT NULL AND header <> ''"

    def get_required_data(self, series: DataFrame) -> Any:
        """Returns the data required by the processing functions"""
        return series['header']

    def check_condition_before_process(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing.
        e.g. data is required to be present for analysis"""
        return bool(data)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!
    procedural_participation_extractor = ProceduralParticipationExtractor(config)
    procedural_participation_extractor.start()
