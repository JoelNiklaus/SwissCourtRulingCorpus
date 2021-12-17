from __future__ import annotations
from typing import Any, TYPE_CHECKING

import pandas as pd
from sqlalchemy.engine.base import Engine
from sqlalchemy.sql.expression import text
from sqlalchemy.sql.schema import MetaData, Table

from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor
from scrc.enums.section import Section
from scrc.utils.main_utils import get_config

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame


class CourtCompositionExtractor(AbstractExtractor):
    """
    Extracts the court composition from the header section. This is part of the judicial person extraction task.
    """

    def __init__(self, config: dict):
        super().__init__(config, function_name='court_composition_extracting_functions', col_name='court_composition')
        self.processed_file_path = self.progress_dir / "spiders_court_composition_extracted.txt"
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

    def get_required_data(self, series: DataFrame) -> Any:
        """Returns the data required by the processing functions"""
        return {Section.HEADER: series['header'],
                Section.FOOTER: series['footer']}
        
    def select_df(self, engine: str, spider: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        return self.select(engine, 'section headersection LEFT JOIN decision on decision.decision_id = headersection.decision_id LEFT JOIN language ON language.language_id = decision.language_id LEFT JOIN file ON file.file_id = decision.file_id LEFT JOIN section footersection on headersection.decision_id = footersection.decision_id and footersection.section_type_id = 6', f"headersection.decision_id, headersection.section_text as header, footersection.section_text as footer,'{spider}' as spider, iso_code as language, html_url", where=f"headersection.section_type_id = 1 AND headersection.decision_id IN (SELECT decision_id from decision WHERE chamber_id IN (SELECT chamber_id FROM chamber WHERE spider_id IN (SELECT spider_id FROM spider WHERE spider.name = '{spider}')))", chunksize=self.chunksize)
    
    def save_data_to_database(self, df: pd.DataFrame, engine: Engine):
        
        key_to_id = {
            'federal_judge': 1,
            'deputy_federal_judge': 2,
            'clerk': 3
        }
        
        for idx, row in df.iterrows():
            with engine.connect() as conn:
                t_person = Table('person', MetaData(), autoload_with=conn)
                t_jud_person = Table('judicial_person', MetaData(), autoload_with=conn)
                
                # Delete person
                stmt = t_jud_person.delete().where(text(f"decision_id in ({','.join([chr(39)+str(item)+chr(39) for item in df['decision_id'].tolist()])})"))
                conn.execute(stmt)
                
                court_composition = row['court_composition']
                president = court_composition['president']
                
                # create president person
                stmt = t_person.insert().returning(text("person_id")).values([{"name": president['name'], "is_natural_person": True, "gender": president['gender']}])
                person_id = conn.execute(stmt).fetchone()['person_id']
                stmt = t_jud_person.insert.values([{"decision_id": str(row['decision_id']), "person_id": person_id, "judicial_person_type_id": 1, "isPresident": True}])
                conn.execute(stmt)
                
                # create all judges
                for judge in court_composition.get('judges'):
                    if judge == president:
                        continue
                    stmt = t_person.insert().returning(text("person_id")).values([{"name": judge['name'], "is_natural_person": True, "gender": judge['gender']}])
                    person_id = conn.execute(stmt).fetchone()['person_id']
                    stmt = t_jud_person.insert().values([{"decision_id": str(row['decision_id']), "person_id": person_id, "judicial_person_type_id": 1, "isPresident": False}])
                    conn.execute(stmt)
                    
                # create all clerks
                for clerk in court_composition.get('clerks'):
                    stmt = t_person.insert().returning(text("person_id")).values([{"name": clerk['name'], "is_natural_person": True, "gender": clerk['gender']}])
                    person_id = conn.execute(stmt).fetchone()['person_id']
                    stmt = t_jud_person.insert().values([{"decision_id": str(row['decision_id']), "person_id": person_id, "judicial_person_type_id": 3, "isPresident": False}])
                    conn.execute(stmt)
        with engine.connect() as conn:
            stmt = t_person.delete().where(text(f"NOT EXISTS (SELECT FROM judicial_person jp WHERE jp.person_id = person.person_id UNION SELECT FROM party WHERE party.person_id = person.person_id)"))
            conn.execute(stmt)
                

    def check_condition_before_process(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing. 
        e.g. data is required to be present for analysis"""
        return bool(data)


if __name__ == '__main__':
    config = get_config()
    court_composition_extractor = CourtCompositionExtractor(config)
    court_composition_extractor.start()
