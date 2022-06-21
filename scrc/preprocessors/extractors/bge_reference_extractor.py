from __future__ import annotations
from typing import Any, TYPE_CHECKING, Union
import bs4
import pandas as pd
from sqlalchemy.engine.base import Engine
from sqlalchemy.sql.expression import text
from sqlalchemy.sql.schema import MetaData, Table

from root import ROOT_DIR
from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor
from scrc.enums.section import Section
from scrc.utils.main_utils import get_config
from scrc.utils.sql_select_utils import delete_stmt_decisions_with_df, join_decision_and_language_on_parameter, \
    join_file_on_decision, where_decisionid_in_list, where_string_spider

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame


class BgeReferenceExtractor(AbstractExtractor):
    """
    Extracts the reference to a bger from bge header section. This is part of the criticality prediction task.
    """

    def __init__(self, config: dict):
        super().__init__(config, function_name='bge_reference_extracting_functions', col_name='bge_reference')
        self.processed_file_path = self.progress_dir / "bge_reference_extracted.txt"
        self.logger_info = {
            'start': 'Started extracting the bge references',
            'finished': 'Finished extracting the bge references',
            'start_spider': 'Started extracting the bge references for spider',
            'finish_spider': 'Finished extracting the bge references for spider',
            'saving': 'Saving chunk of bge references',
            'processing_one': 'Extracting the bge references from',
            'no_functions': 'Not extracting the bge references'
        }

    def get_database_selection_string(self, spider: str, lang: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        return f"spider='{spider}' AND header IS NOT NULL AND header <> ''"

    def get_required_data(self, series: pd.DataFrame) -> Union[bs4.BeautifulSoup, str, None]:
        """Returns the data required by the processing functions"""
        return {Section.HEADER: series['header']}

    def select_df(self, engine: str, spider: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        only_given_decision_ids_string = f" AND {where_decisionid_in_list(self.decision_ids)}" if self.decision_ids is not None else ""
        # TODO test what is selected
        # select analog zur Abfrage in court_composition_extractor aber ohne footer
        to_return = self.select(engine, f"section headersection {join_decision_and_language_on_parameter('decision_id', 'headersection.decision_id')} {join_file_on_decision()}", f"headersection.decision_id, headersection.section_text as header, '{spider}' as spider, iso_code as language, html_url", where=f"headersection.section_type_id = 1 AND headersection.decision_id IN {where_string_spider('decision_id', spider)} {only_given_decision_ids_string}", chunksize=self.chunksize)
        return to_return


    def save_data_to_database(self, df: pd.DataFrame, engine: Engine):
        self.logger.info("save data in progress")
        processed_file_path = ROOT_DIR / 'data' / 'progress' / "bge_references_found.txt"
        for _, row in df.iterrows():
            with processed_file_path.open("a") as f:
                if 'bge_reference' in row and row['bge_reference'] != 'no reference found':
                    bge_reference = str(row['bge_reference'])
                    f.write(f"{bge_reference}\n")


    def check_condition_before_process(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing.
        e.g. data is required to be present for analysis"""
        return bool(data)


if __name__ == '__main__':
    config = get_config()
    bge_reference_extractor = BgeReferenceExtractor(config)
    bge_reference_extractor.start()
    # delete bge_reference_found.txt file before letting it run.
