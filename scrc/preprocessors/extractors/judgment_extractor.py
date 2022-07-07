from __future__ import annotations
import configparser
import datetime
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd
from sqlalchemy.engine.base import Engine
from sqlalchemy.sql.expression import text
from sqlalchemy.sql.schema import MetaData, Table
from scrc.enums.judgment import Judgment
from scrc.preprocessors.abstract_preprocessor import AbstractPreprocessor

from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor
from root import ROOT_DIR
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config
from scrc.utils.sql_select_utils import delete_stmt_decisions_with_df, get_judgment_query, get_total_judgments, join_decision_and_language_on_parameter, \
    join_file_on_decision, where_decisionid_in_list, where_string_spider

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame


class JudgmentExtractor(AbstractExtractor):
    """
    Extracts the judgments from the rulings section. This represents the judgment extraction task.
    """

    def __init__(self, config: dict):
        super().__init__(config, function_name='judgment_extracting_functions', col_name='judgments')
        self.logger = get_logger(__name__)
        self.processed_file_path = self.progress_dir / "spiders_judgment_extracted.txt"
        self.logger_info = {
            'start': 'Started extracting judgments',
            'finished': 'Finished extracting judgments',
            'start_spider': 'Started extracting the judgments for spider',
            'finish_spider': 'Finished extracting the judgments for spider',
            'saving': 'Saving chunk of judgments',
            'processing_one': 'Extracting the judgment from',
            'no_functions': 'Not extracting the judgments.'
        }

    def get_database_selection_string(self, spider: str, lang: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        return f"spider='{spider}' AND rulings IS NOT NULL AND rulings <> ''"
    
    def get_coverage(self, spider: str):
        with self.get_engine(self.db_scrc).connect() as conn:
            total_judgments = conn.execute(get_total_judgments(spider)).fetchone()
            coverage_result = conn.execute(get_judgment_query(spider)).fetchone()
            coverage =  round(coverage_result[0] / total_judgments[0]  * 100, 2)
            self.logger.info(f'{spider}: Found judgment outcome for {coverage}% of the rulings')



    def get_required_data(self, series: DataFrame) -> Any:
        """Returns the data required by the processing functions"""
        return series['section_text']

    def select_df(self, engine: str, spider: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        only_given_decision_ids_string = f" AND {where_decisionid_in_list(self.decision_ids)}" if self.decision_ids is not None else ""
        return self.select(engine,
                           f"section {join_decision_and_language_on_parameter('decision_id', 'section.decision_id')} {join_file_on_decision()}",
                           f"section.decision_id, section_text, '{spider}' as spider, iso_code as language, html_url",
                           where=f"section.section_type_id = 5 AND section.decision_id IN {where_string_spider('decision_id', spider)} {only_given_decision_ids_string}",
                           chunksize=self.chunksize)

    def save_data_to_database(self, df: pd.DataFrame, engine: Engine):
        if not AbstractPreprocessor._check_write_privilege(engine):
            AbstractPreprocessor.create_dir(self.output_dir, os.getlogin())
            path = Path.joinpath(self.output_dir, os.getlogin(), datetime.now().isoformat() + '.json')

            with path.open("a") as f:
                df.to_json(f, default_handler=str)
            return
        with engine.connect() as conn:
            t = Table('judgment_map', MetaData(), autoload_with=engine)
            stmt = t.delete().where(delete_stmt_decisions_with_df(df))
            conn.execute(stmt)
            for idx, row in df.iterrows():
                # print(df.loc[:, df.columns!= 'section_text'])
                if row['judgments']:  # only insert, when we find judgments
                    # Delete and reinsert as no upsert command is available
                    for k in row['judgments']:
                        judgment_type_id = Judgment(k).value
                        stmt = t.insert().values([{"decision_id": str(row['decision_id']),
                                                   "judgment_id": judgment_type_id}])
                        conn.execute(stmt)
                else:
                    self.logger.warning(f"No judgments found for {row['html_url']}")

    def check_condition_before_process(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing.
        e.g. data is required to be present for analysis"""
        return bool(data)


if __name__ == '__main__':
    config = get_config()
    judgment_extractor = JudgmentExtractor(config)
    judgment_extractor.start()
