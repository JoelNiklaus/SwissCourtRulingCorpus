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
from scrc.utils.sql_select_utils import delete_stmt_decisions_with_df, join_decision_and_language_on_parameter, \
    join_file_on_decision, where_decisionid_in_list, where_string_spider

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

    def get_coverage(self, engine: Engine, spider: str):
         """No coverage implemented"""

    def get_database_selection_string(self, spider: str, lang: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        return f"spider='{spider}' AND header IS NOT NULL AND header <> ''"

    def get_required_data(self, series: DataFrame) -> Any:
        """Returns the data required by the processing functions"""
        return {Section.HEADER: series['header'], Section.FACTS: series['facts'],
                Section.CONSIDERATIONS: series['considerations'], Section.RULINGS: series['rulings'],
                Section.FOOTER: series['footer']}
        
    def get_coverage(self, spider: str):
        pass
            
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
        only_given_decision_ids_string = f" AND {where_decisionid_in_list(self.decision_ids)}" if self.decision_ids is not None else ""
        # Joining the footer on to the same row as the header so each decision goes through the erxtractor only once per decision
        section_self_join = 'LEFT JOIN section footersection on headersection.decision_id = footersection.decision_id ' \
                            'and footersection.section_type_id = 6'
        header_section_join = join_decision_and_language_on_parameter('decision_id', 'headersection.decision_id')
        tables = f"section headersection {header_section_join} {join_file_on_decision()} {section_self_join}"
        columns = f"headersection.decision_id, headersection.section_text as header, footersection.section_text as footer, " \
                  f"'{spider}' as spider, iso_code as language, html_url"
        where = f"headersection.section_type_id = 1 " \
                f"AND headersection.decision_id IN {where_string_spider('decision_id', spider)} {only_given_decision_ids_string}"
        return self.select(engine, tables, columns, where=where, chunksize=self.chunksize)

    def save_data_to_database(self, df: pd.DataFrame, engine: Engine):
        """
            'plaintiff': 1,
            'defendant': 2,
            'representation_plaintiff': 3,
            'representation_defendant': 4
        """
        with engine.connect() as conn:
            t_person = Table('person', MetaData(), autoload_with=conn)
            t_party = Table('party', MetaData(), autoload_with=conn)

            # Delete person
            # Delete and reinsert as no upsert command is available
            stmt = t_party.delete().where(delete_stmt_decisions_with_df(df))
            conn.execute(stmt)
            
            for _, row in df.iterrows():
                if not row['parties']: continue
                parties = json.loads(row['parties'])

                # create all plaintiffs
                offset = 0
                for plaintiff in parties.get('plaintiffs'):
                    # Insert person into the person table
                    plaintiff_dict = {"name": plaintiff['name'].strip(),
                                      "is_natural_person": plaintiff['legal_type'] == 'natural person',
                                      "gender": plaintiff['gender']}
                    stmt = t_person.insert().returning(column("person_id")).values([plaintiff_dict])
                    person_id = conn.execute(stmt).fetchone()['person_id']
                    # Then insert the person into the party table
                    stmt = t_party.insert().values(
                        [{"decision_id": str(row['decision_id']), "person_id": person_id, "party_type_id": 1 + offset}])
                    conn.execute(stmt)
                    for representant in plaintiff.get('legal_counsel'):
                        if not 'name' in representant or not representant['name']: continue
                        # Insert their representation into the person and party tables
                        representant_dict = {"name": representant['name'].strip(),
                                             "is_natural_person": representant['legal_type'] == 'natural person',
                                             "gender": representant['gender']}
                        stmt = t_person.insert().returning(text("person_id")).values([representant_dict])
                        person_id = conn.execute(stmt).fetchone()['person_id']
                        stmt = t_party.insert().values([{"decision_id": str(row['decision_id']), "person_id": person_id,
                                                         "party_type_id": 3 + offset}])
                        conn.execute(stmt)

                # create all defendants
                offset = 1
                for defendant in parties.get('defendants'):
                    # Insert person into the person table
                    defendant_dict = {"name": plaintiff['name'].strip(),
                                      "is_natural_person": plaintiff['legal_type'] == 'natural person',
                                      "gender": plaintiff['gender']}
                    stmt = t_person.insert().returning(text("person_id")).values([defendant_dict])
                    person_id = conn.execute(stmt).fetchone()['person_id']
                    # Then insert the person into the party table
                    stmt = t_party.insert().values(
                        [{"decision_id": str(row['decision_id']), "person_id": person_id, "party_type_id": 1 + offset}])
                    conn.execute(stmt)
                    for representant in defendant.get('legal_counsel'):
                        # Insert their representation into the person and party tables
                        representant_dict = {"name": representant['name'].strip(),
                                             "is_natural_person": representant['legal_type'] == 'natural person',
                                             "gender": representant['gender']}
                        stmt = t_person.insert().returning(text("person_id")).values([representant_dict])
                        person_id = conn.execute(stmt).fetchone()['person_id']
                        stmt = t_party.insert().values([{"decision_id": str(row['decision_id']), "person_id": person_id,
                                                         "party_type_id": 3 + offset}])
                        conn.execute(stmt)

        with engine.connect() as conn:
            stmt = t_person.delete().where(text(
                f"NOT EXISTS (SELECT FROM judicial_person jp WHERE jp.person_id = person.person_id UNION SELECT FROM party WHERE party.person_id = person.person_id)"))
            conn.execute(stmt)


if __name__ == '__main__':
    config = get_config()
    procedural_participation_extractor = ProceduralParticipationExtractor(config)
    procedural_participation_extractor.start()
