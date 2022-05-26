from __future__ import annotations
from typing import Any, TYPE_CHECKING

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
        only_given_decision_ids_string = f" AND {where_decisionid_in_list(self.decision_ids)}" if self.decision_ids is not None else ""
        # Joining the footer on to the same row as the header so each decision goes through the erxtractor only once per decision
        section_self_join = 'LEFT JOIN section footersection ON headersection.decision_id = footersection.decision_id ' \
                            'AND footersection.section_type_id = 6'
        decision_language_join = join_decision_and_language_on_parameter('decision_id', 'headersection.decision_id')
        tables = f"section headersection {decision_language_join} {join_file_on_decision()} {section_self_join}"
        columns = f"headersection.decision_id, headersection.section_text as header, " \
                  f"footersection.section_text as footer,'{spider}' as spider, iso_code as language, html_url"
        where = f"headersection.section_type_id = 1 " \
                f"AND headersection.decision_id IN {where_string_spider('decision_id', spider)} {only_given_decision_ids_string}"
        return self.select(engine, tables, columns, where=where, chunksize=self.chunksize)

    def save_data_to_database(self, df: pd.DataFrame, engine: Engine):
        with engine.connect() as conn:
            t_person = Table('person', MetaData(), autoload_with=conn)
            t_jud_person = Table('judicial_person', MetaData(), autoload_with=conn)
            for idx, row in df.iterrows():
                # Delete person
                # Delete and reinsert as no upsert command is available
                stmt = t_jud_person.delete().where(delete_stmt_decisions_with_df(df))
                conn.execute(stmt)

                court_composition = row['court_composition']
                president = court_composition['president']

                # create president person
                president_dict = {"name": president['name'].strip(),
                                  "is_natural_person": True,
                                  "gender": president['gender']}
                stmt = t_person.insert().returning(text("person_id")).values(
                    [president_dict])
                person_id = conn.execute(stmt).fetchone()['person_id']
                person_dict = {"decision_id": str(row['decision_id']), "person_id": person_id,
                               "judicial_person_type_id": 1, "isPresident": True}
                stmt = t_jud_person.insert.values([person_dict])
                conn.execute(stmt)

                # create all judges
                for judge in court_composition.get('judges'):
                    if judge == president:
                        continue
                    judge_dict = {"name": judge['name'].strip(), "is_natural_person": True, "gender": judge['gender']}
                    stmt = t_person.insert().returning(text("person_id")).values([judge_dict])
                    person_id = conn.execute(stmt).fetchone()['person_id']
                    person_dict = {"decision_id": str(row['decision_id']),
                                   "person_id": person_id, "judicial_person_type_id": 1,
                                   "isPresident": False}
                    stmt = t_jud_person.insert().values([person_dict])
                    conn.execute(stmt)

                # create all clerks
                for clerk in court_composition.get('clerks'):
                    clerk_dict = {"name": clerk['name'].strip(), "is_natural_person": True, "gender": clerk['gender']}
                    stmt = t_person.insert().returning(text("person_id")).values([clerk_dict])
                    person_id = conn.execute(stmt).fetchone()['person_id']
                    person_dict = {"decision_id": str(row['decision_id']),
                                   "person_id": person_id, "judicial_person_type_id": 2,
                                   "isPresident": False}
                    stmt = t_jud_person.insert().values([person_dict])
                    conn.execute(stmt)

            where = f"NOT EXISTS (SELECT FROM judicial_person jp WHERE jp.person_id = person.person_id " \
                    f"UNION SELECT FROM party WHERE party.person_id = person.person_id)"
            stmt = t_person.delete().where(text(where))
            conn.execute(stmt)

    def check_condition_before_process(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing. 
        e.g. data is required to be present for analysis"""
        return bool(data)


if __name__ == '__main__':
    config = get_config()
    court_composition_extractor = CourtCompositionExtractor(config)
    court_composition_extractor.start()
