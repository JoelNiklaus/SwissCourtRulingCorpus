from __future__ import annotations
import json
from typing import Any, TYPE_CHECKING

import pandas as pd
from sqlalchemy.engine.base import Engine
from sqlalchemy.sql.expression import column, text
from sqlalchemy.sql.schema import MetaData, Table

from scrc.enums.section import Section
from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor
from scrc.utils.main_utils import get_config

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame


class ProceduralParticipationExtractor(AbstractExtractor):
    """
    Extracts the parties from the header section. This is part of the judicial person extraction task.
    """

    def __init__(self, config: dict):
        super().__init__(config, function_name='procedural_participation_extracting_functions', col_name='parties')
        self.processed_file_path = self.progress_dir / "spiders_procedural_participation_extracted.txt"
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
        return {Section.HEADER: series['header'], Section.FACTS: series['facts'],
                Section.CONSIDERATIONS: series['considerations'], Section.RULINGS: series['rulings'],
                Section.FOOTER: series['footer']}

    def check_condition_before_process(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing.
        e.g. data is required to be present for analysis"""
        return bool(data)
    
    def get_required_data(self, series: DataFrame) -> Any:
        """Returns the data required by the processing functions"""
        return {Section.HEADER: series['header'],
                Section.FOOTER: series['footer']}
    
    def select_df(self, engine: str, spider: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        return self.select(engine, 'section headersection LEFT JOIN decision on decision.decision_id = headersection.decision_id LEFT JOIN language ON language.language_id = decision.language_id LEFT JOIN file ON file.file_id = decision.file_id LEFT JOIN section footersection on headersection.decision_id = footersection.decision_id and footersection.section_type_id = 6', f"headersection.decision_id, headersection.section_text as header, footersection.section_text as footer,'{spider}' as spider, iso_code as language, html_url", where=f"headersection.section_type_id = 1 AND headersection.decision_id IN (SELECT decision_id from decision WHERE chamber_id IN (SELECT chamber_id FROM chamber WHERE spider_id IN (SELECT spider_id FROM spider WHERE spider.name = '{spider}')))", chunksize=self.chunksize)

    def save_data_to_database(self, df: pd.DataFrame, engine: Engine):
        
        """
            'plaintiff': 1,
            'defendant': 2,
            'representation_plaintiff': 3,
            'representation_defendant': 4
        """
        
        for idx, row in df.iterrows():
            with engine.connect() as conn:
                t_person = Table('person', MetaData(), autoload_with=conn)
                t_party = Table('party', MetaData(), autoload_with=conn)
                
                # Delete person
                stmt = t_party.delete().where(text(f"decision_id in ({','.join([chr(39)+str(item)+chr(39) for item in df['decision_id'].tolist()])})"))
                conn.execute(stmt)
                
                parties = json.loads(row['parties'])
                
                # create all plaintiffs
                offset = 0
                for plaintiff in parties.get('plaintiffs'):
                    stmt = t_person.insert().returning(column("person_id")).values([{"name": plaintiff['name'], "is_natural_person": plaintiff['legal_type'] == 'natural person', "gender": plaintiff['gender']}])
                    person_id = conn.execute(stmt).fetchone()['person_id']
                    stmt = t_party.insert().values([{"decision_id": str(row['decision_id']), "person_id": person_id, "party_type_id": 1+offset}])
                    conn.execute(stmt)
                    for representant in plaintiff.get('legal_counsel'):
                        stmt = t_person.insert().returning(text("person_id")).values([{"name": representant['name'], "is_natural_person": representant['legal_type'] == 'natural person', "gender": representant['gender']}])
                        person_id = conn.execute(stmt).fetchone()['person_id']
                        stmt = t_party.insert().values([{"decision_id": str(row['decision_id']), "person_id": person_id, "party_type_id": 3+offset}])
                        conn.execute(stmt)
                    
                # create all defendants
                offset = 1
                for defendant in parties.get('defendants'):
                    stmt = t_person.insert().returning(text("person_id")).values([{"name": plaintiff['name'], "is_natural_person": plaintiff['legal_type'] == 'natural person', "gender": plaintiff['gender']}])
                    person_id = conn.execute(stmt).fetchone()['person_id']
                    stmt = t_party.insert().values([{"decision_id": str(row['decision_id']), "person_id": person_id, "party_type_id": 1+offset}])
                    conn.execute(stmt)
                    for representant in defendant.get('legal_counsel'):
                        stmt = t_person.insert().returning(text("person_id")).values([{"name": representant['name'], "is_natural_person": representant['legal_type'] == 'natural person', "gender": representant['gender']}])
                        person_id = conn.execute(stmt).fetchone()['person_id']
                        stmt = t_party.insert().values([{"decision_id": str(row['decision_id']), "person_id": person_id, "party_type_id": 3+offset}])
                        conn.execute(stmt)
                    
        with engine.connect() as conn:
            stmt = t_person.delete().where(text(f"NOT EXISTS (SELECT FROM judicial_person jp WHERE jp.person_id = person.person_id UNION SELECT FROM party WHERE party.person_id = person.person_id)"))
            conn.execute(stmt)
if __name__ == '__main__':
    config = get_config()
    procedural_participation_extractor = ProceduralParticipationExtractor(config)
    procedural_participation_extractor.start()
