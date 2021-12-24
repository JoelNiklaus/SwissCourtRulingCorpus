from __future__ import annotations
from typing import Any, TYPE_CHECKING
from sqlalchemy.sql.expression import text

import pandas as pd
from sqlalchemy.engine.base import Engine
from sqlalchemy.sql.schema import MetaData, Table
from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config
from scrc.utils.sql_select_utils import delete_stmt_decisions_with_df, join_decision_and_language_on_parameter, join_file_on_decision, where_string_spider

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
        self.processed_file_path = self.progress_dir / "spiders_lower_court_extracted.txt"
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
        """Returns the data required by the processing functions"""
        return series['section_text']

    def check_condition_before_process(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing.
        e.g. data is required to be present for analysis"""
        return bool(data)

    def select_df(self, engine: str, spider: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        return self.select(engine, f"section {join_decision_and_language_on_parameter('decision_id', 'section.decision_id')} {join_file_on_decision()}", f"section.decision_id, section_text, '{spider}' as spider, iso_code as language, html_url", where=f"section.section_type_id = 2 AND section.decision_id IN {where_string_spider('decision_id', spider)}", chunksize=self.chunksize)
    
    def save_data_to_database(self, df: pd.DataFrame, engine: Engine):
        for idx, row in df.iterrows():
            if not 'lower_court' in row or row['lower_court'] is None:
                continue
            lower_court = row["lower_court"]
            res = {}
            
            if 'court' in lower_court and lower_court['court'] is not None:
                res['court_id'] = list(self.select(engine, 'court', 'court_id', f"court_string = '{lower_court['court']}'"))[0]['court_id'][0]
                res['court_id'] = int(res['court_id']) if res['court_id'] is not None else None
            if 'canton' in lower_court and lower_court['canton'] is not None:
                res['canton_id'] = list(self.select(engine, 'canton', 'canton_id', f"short_code = '{lower_court['canton']}'"))[0]['canton_id'][0]
                res['canton_id'] = int(res['canton_id']) if res['canton_id'] is not None else None
            if 'chamber' in lower_court and lower_court['chamber'] is not None:
                res['chamber_id'] = list(self.select(engine, 'chamber', 'chamber_id', f"chamber_string = '{lower_court['chamber']}'"))[0]['chamber_id'][0]
                res['chamber_id'] = int(res['chamber_id']) if res['chamber_id'] is not None else None
                
            with engine.connect() as conn:
                t = Table('lower_court', MetaData(), autoload_with=conn)
                stmt = t.delete().where(delete_stmt_decisions_with_df(df))
                conn.execute(stmt)
                stmt = t.insert().values([{"decision_id": str(row['decision_id']), "court_id": res.get('court_id'), "canton_id": res.get('canton_id'), "chamber_id": res.get('chamber_id'), "date": lower_court.get('date'), "file_number": lower_court.get('file_number')}])
                conn.execute(stmt)
                

if __name__ == '__main__':
    config = get_config()

    lower_court_extractor = LowerCourtExtractor(config)
    lower_court_extractor.start()
