from __future__ import annotations
from typing import Any, TYPE_CHECKING, Union
import bs4
import pandas as pd
from sqlalchemy.engine.base import Engine
from sqlalchemy.sql.expression import text
from sqlalchemy.sql.schema import MetaData, Table

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
            'start': 'Started extracting the court compositions',
            'finished': 'Finished extracting the court compositions',
            'start_spider': 'Started extracting the court compositions for spider',
            'finish_spider': 'Finished extracting the court compositions for spider',
            'saving': 'Saving chunk of court compositions',
            'processing_one': 'Extracting the court composition from',
            'no_functions': 'Not extracting the court compositions.'
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
        # select analog zur Abfrage in court_composition_extractor aber ohne footer
        return self.select(engine, f"section headersection {join_decision_and_language_on_parameter('decision_id', 'headersection.decision_id')} {join_file_on_decision()}", f"headersection.decision_id, headersection.section_text as header, '{spider}' as spider, iso_code as language, html_url", where=f"headersection.section_type_id = 1 AND headersection.decision_id IN {where_string_spider('decision_id', spider)} {only_given_decision_ids_string}", chunksize=self.chunksize)

    def save_data_to_database(self, df: pd.DataFrame, engine: Engine):
        # TODO add table to postgres
       """
        for idx, row in df.iterrows():
            bge_reference = row['bge_reference']
            with engine.connect() as conn:
                # TODO first add bge_reference in database
                t = Table('bge_reference', MetaData(), autoload_with=conn)
                # delete and reinsert
                stmt = t.delete().where(delete_stmt_decisions_with_df(df))
                conn.execute(stmt)
                stmt = t.insert().values({"decision_id": str(row['decision_id']), "bge_reference": str(row['bge_reference'])})
                conn.execute(stmt)
"""

    def check_condition_before_process(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing.
        e.g. data is required to be present for analysis"""
        return bool(data)


if __name__ == '__main__':
    config = get_config()
    bge_reference_extractor = BgeReferenceExtractor(config)
    bge_reference_extractor.start()
