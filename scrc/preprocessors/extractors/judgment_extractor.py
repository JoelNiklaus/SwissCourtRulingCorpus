from __future__ import annotations
import configparser
from typing import Any, TYPE_CHECKING

import pandas as pd
from sqlalchemy.engine.base import Engine
from sqlalchemy.sql.expression import text
from sqlalchemy.sql.schema import MetaData, Table

from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor
from root import ROOT_DIR
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config

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

    def get_required_data(self, series: DataFrame) -> Any:
        """Returns the data required by the processing functions"""
        return series['section_text']
    
    def select_df(self, engine: str, spider: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        return self.select(engine, 'section LEFT JOIN decision on decision.decision_id = section.decision_id LEFT JOIN language ON language.language_id = decision.language_id LEFT JOIN file ON file.file_id = decision.file_id', f"section.decision_id, section_text, '{spider}' as spider, iso_code as language, html_url", where=f"section.section_type_id = 5 AND section.decision_id IN (SELECT decision_id from decision WHERE chamber_id IN (SELECT chamber_id FROM chamber WHERE spider_id IN (SELECT spider_id FROM spider WHERE spider.name = '{spider}')))", chunksize=self.chunksize)

    def save_data_to_database(self, df: pd.DataFrame, engine: Engine):
        
        key_to_id = {
            'approval': 1,
            'dismissal': 2,
            'inadmissible': 3,
            'partial_approval': 4,
            'partial_dismissal': 5,
            'unification': 6,
            'write_off': 7
        }
        
        for idx, row in df.iterrows():
            with engine.connect() as conn:
                t = Table('judgment_map', MetaData(), autoload_with=engine)
                stmt = t.delete().where(text(f"decision_id in ({','.join([chr(39)+str(item)+chr(39) for item in df['decision_id'].tolist()])})"))
                engine.execute(stmt)
                for k in row['judgments']: 
                    judgment_type_id = key_to_id[k]
                    stmt = t.insert().values([{"decision_id": str(row['decision_id']), "judgment_id": judgment_type_id}])
                    engine.execute(stmt)
    
    def check_condition_before_process(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing.
        e.g. data is required to be present for analysis"""
        return bool(data)


if __name__ == '__main__':
    config = get_config()

    judgment_extractor = JudgmentExtractor(config)
    judgment_extractor.start()
