from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional
import numpy as np
import pandas as pd
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.sql.expression import text
from sqlalchemy.sql.schema import MetaData, Table
from transformers.file_utils import add_code_sample_docstrings

from scrc.preprocessors.abstract_preprocessor import AbstractPreprocessor

if TYPE_CHECKING:
    from sqlalchemy.engine.base import Engine

def join_decision_on_language() -> str:
    """Returns the join string for the language join. The decision table has to be joined already.

    Returns:
        str: The Join string
    """
    return ' LEFT JOIN language ON language.language_id = decision.language_id '

def join_decision_on_parameter(decision_field: str, target_table_and_field: str) -> str:
    """Join the decision table on the decision field and specified table and target string. 
        ('file_id', 'file.file_id') returns 'LEFT JOIN decision on decision.file_id = file.file_id'

    Args:
        decision_field (str): the fieldname on the decision table. Most likely `decision_id` or `file_id`
        target_table_and_field (str): the target of the join in the form of `<TABLE>.<FIELD>`

    Returns:
        str: The join string
    """
    return f" LEFT JOIN decision on decision.{decision_field}={target_table_and_field} "

def join_decision_and_language_on_parameter(decision_field, target_table_and_field) -> str:
    """Join the decision table on the decision field and specified table and target string and then joins the language table. 
        ('file_id', 'file.file_id') returns 'LEFT JOIN decision on decision.file_id = file.file_id LEFT JOIN language ON language.language_id = decision.language_id'

    Args:
        decision_field (str): the fieldname on the decision table. Most likely `decision_id` or `file_id`
        target_table_and_field (str): the target of the join in the form of `<TABLE>.<FIELD>`

    Returns:
        str: The join string
    """
    return join_decision_on_parameter(decision_field, target_table_and_field) + join_decision_on_language()

def join_file_on_decision() -> str:
    """Returns the join string for the foining of the file table onto the decision table

    Returns:
        str: The join string
    """
    return ' LEFT JOIN file ON file.file_id = decision.file_id '

def where_string_spider(decision_field: str, spider: str) -> str:
    """Returns the string for the where clause in the sql selection such that only the decisions of certain spider are selected.
        Use by <TABLE>.<FIELDNAME> IN where_string_spider(<FIELDNAME>, <SPIDER>)
    Args:
        decision_field (str): The field name to be searched in the decision table
        spider (str): The spider name

    Returns:
        str: The where clause
    """
    return f" (SELECT {decision_field} from decision WHERE chamber_id IN (SELECT chamber_id FROM chamber WHERE spider_id IN (SELECT spider_id FROM spider WHERE spider.name = '{spider}'))) "

def save_from_text_to_database(engine: Engine, df: pd.DataFrame):
    """ Saving these fields
            Column('language', String),
            Column('chamber', String),
            Column('date', Date),
            Column('file_name', String),
            Column('file_number', String),
            Column('file_number_additional', String),
            Column('html_url', String),
            Column('html_raw', String),
            Column('pdf_url', String),
            Column('pdf_raw', String),
    """
    def save_to_db(df: pd.DataFrame, table: str):
        df.to_sql(table, engine, if_exists="append", index=False)
        
    def add_ids_to_df_for_decision(series: pd.DataFrame) -> pd.DataFrame:
        series['file_id'] = pd.read_sql(f"SELECT file_id FROM file WHERE file_name = '{series['file_name']}'", engine.connect())["file_id"][0]
        series['language_id'] = -1
        chamber_id = pd.read_sql(f"SELECT chamber_id FROM chamber WHERE chamber_string = '{series['chamber']}'", engine.connect())['chamber_id']
        if len(chamber_id) == 0:
            raise ValueError(f"The chamber {series['chamber']} was not found in the database. Add it with the respective court and spider")
        else:
            series['chamber_id'] = chamber_id[0]
            
        # TODO: Add topic recognition, similar to the title of the court decision
        series['topic'] = '' 
        return series
        
    def save_the_file_numbers(series: pd.DataFrame) -> pd.DataFrame:
        series['decision_id'] = pd.read_sql(f"SELECT decision_id FROM decision WHERE file_id = '{series['file_id']}'", engine.connect())["decision_id"][0]
        with engine.connect as conn:
             t = Table('file_number', MetaData(), autoload_with=engine)
             # Delete and reinsert as no upsert command is available
             stmt = t.delete().where(delete_stmt_decisions_with_df(series))
             conn.execute(stmt)
        series['decision_id'] = pd.read_sql(f"SELECT decision_id FROM decision WHERE file_id = '{series['file_id']}'", engine.connect())["decision_id"][0]
        series['text'] = series['file_number']
        save_to_db([['decision_id', 'text']], 'file_number')
        if ('file_number_additional' in series and series['file_number_additional'] is not None and len(series['file_number_additional'])>0):
            series['text'] = series['file_number_additional']
            save_to_db([['decision_id', 'text']], 'file_number')
        return series
        
    # Delete old decision and file entries
    with engine.connect() as conn: 
        t_fil = Table('file', MetaData(), autoload_with=engine)
        t_dec = Table('decision', MetaData(), autoload_with=engine)
        file_name_list = ','.join(["'" + str(item)+"'" for item in df['file_name'].tolist()])
        stmt = t_fil.select().where(text(f"file_name in ({file_name_list})"))
        file_ids = [item['file_id'] for item in conn.execute(stmt).all()]
        if len(file_ids) > 0:
            file_ids_list = ','.join(["'" + str(item)+"'" for item in file_ids])
            # decision_ids = [item['decision_id'] for item in conn.execute(t_dec.select().where(text(f"file_id in ({file_ids_list})"))).all()]
            
            stmt = t_dec.delete().where(text(f"file_id in ({file_ids_list})"))
            conn.execute(stmt)
            stmt = t_fil.delete().where(text(f"file_id in ({file_ids_list})"))
            conn.execute(stmt)
            
                
    
    save_to_db(df[['file_name', 'html_url', 'pdf_url', 'html_raw', 'pdf_raw']], 'file')
    
    df.apply(add_ids_to_df_for_decision, 1)
    save_to_db(df[['language_id', 'chamber_id', 'file_id', 'date', 'topic']], 'decision')
    df.apply(save_the_file_numbers, 1)
    
    
def delete_stmt_decisions_with_df(df: pd.DataFrame) -> TextClause:
    decision_id_list = ','.join(["'" + str(item)+"'" for item in df['decision_id'].tolist()])
    return text(f"decision_id in ({decision_id_list})")

def join(table_name: str, join_field: str = 'decision_id', join_table: str = 'd') -> str:
    """ Helper function """
    return f" LEFT JOIN {table_name} ON {table_name}.{join_field} = {join_table}.{join_field} "
    
def map_join(map_field: str, new_map_field_name: str, table: str, fill: Dict[str, str], group: str = 'decision_id', join_table: str = 'd', additional_fields: str = '') -> str:
    """ Joins a table and concatenates multiple value onto one line """
    if fill:
        return  (f" LEFT JOIN (SELECT {table}_mapped.{group}, array_agg(({fill.get('field_name')})) {new_map_field_name} FROM (SELECT {fill.get('field_name')}, {group} FROM {table} LEFT JOIN {fill.get('table_name')} "
                f" ON {fill.get('table_name')}.{fill.get('join_field')} = {table}.{fill.get('join_field')}) as {table}_mapped GROUP BY {group}) as {table} ON {table}.{group} = {join_table}.{group} ")
        
    return f" LEFT JOIN (SELECT {table}.{group}, array_agg({table}.{map_field}) {new_map_field_name} {additional_fields} FROM {table} GROUP BY {group}) as {table} ON {table}.{group} = {join_table}.{group}"

def join_decision_with_everything_joined() -> str:
    """Join every table onto decision except party, judicial_person person, party_type, judicial_person_type, paragraph

    Returns:
        str: The join string
    """
    tables = "decision d"
    tables += join('file', 'file_id')
    tables += join('section') + ' LEFT JOIN section_type ON section_type.section_type_id = section.section_type_id'
    tables += join('lower_court')
    tables += join('language', 'language_id')
    tables += join('chamber', 'chamber_id') + ' LEFT JOIN court ON court.court_id = chamber.court_id LEFT JOIN spider ON chamber.spider_id = spider.spider_id'
    tables += map_join('citation_id', 'citations', 'citation', fill = {'table_name': 'citation_type', 'field_name': 'name, text, url', 'join_field': 'citation_type_id'})
    tables += map_join('judgment_id', 'judgments', 'judgment_map', fill = {'table_name': 'judgment', 'field_name': 'text', 'join_field': 'judgment_id'})
    tables += map_join('text', 'file_numbers', 'file_number')
    
        
    return  tables

    
def join_tables_on_decision(tables: List[str]) -> str:
    """Usage SELECT <FIELDS YOU WANT, PREFIXED WITH TABLENAME> FROM f"join_tables_on_decision(['TABLE_1', 'TABLE_2'])"

    Args:
        tables (List[str]): [description]

    Returns:
        str: [description]
    """
    join_string = 'decision d'
    
    
    
    if ('file' in tables):
        join_string += join('file', 'file_id')
        
    if ('section' in tables or 'section_type' in tables or 'paragraph' in tables):
        join_string += join('section') + ' LEFT JOIN section_type ON section_type.section_type_id = section.section_type_id'
        
    if ('lower_court' in tables):
        join_string += join('lower_court')
        
    if ('language' in tables):
        join_string += join('language', 'language_id')
        
    if ('chamber' in tables or 'court' in tables or 'spider' in tables):
        join_string += join('chamber', 'chamber_id') + ' LEFT JOIN court ON court.court_id = chamber.court_id LEFT JOIN spider ON chamber.spider_id = spider.spider_id'
        
    if ('citation' in tables or 'citation_type' in tables):
        join_string += map_join('citation_id', 'citations', 'citation', fill = {'table_name': 'citation_type', 'field_name': 'name, text, url', 'join_field': 'citation_type_id'})
        
    if ('judgment_map' in tables or 'judgment' in tables):
        join_string += map_join('judgment_id', 'judgments', 'judgment_map', fill = {'table_name': 'judgment', 'field_name': 'text', 'join_field': 'judgment_id'})
        
    if ('file_number' in tables):
        tables += map_join('text', 'file_numbers', 'file_number')
        
    if ('paragraph' in tables):
        tables += map_join('paragraph_id', 'paragraphs', 'paragraph')
    
    return join_string

def select_paragraphs_with_decision_and_meta_data() -> str:
    """ 
        Edit this according to the example given below. 
        Easiest function to default join tables to a decision.
    """
    return join_tables_on_decision(['citation'])

def select_fields_from_table(fields: List[str], table): 
    fields_strings = [f"{table}.{field}" for field in fields]
    return ", ".join(fields_strings)

def where_decisionid_in_list(decision_ids):
    decision_id_string = ','.join(["'" + str(item)+"'" for item in decision_ids])
    return f"decision_id IN ({decision_id_string})"

def convert_to_binary_judgments(df, with_partials=False, with_write_off=False, with_unification=False,
                            with_inadmissible=False, make_single_label=True):
    def clean(judgments):
        out = set()
        for judgment in judgments:
            # remove "partial_" from all the items to merge them with full ones
            if not with_partials:
                judgment = judgment.replace("partial_", "")

            out.add(judgment)

        if not with_write_off:
            # remove write_off because reason for it happens mostly behind the scenes and not written in the facts
            if 'write_off' in judgments:
                out.remove('write_off')

        if not with_unification:
            # remove unification because reason for it happens mostly behind the scenes and not written in the facts
            if 'unification' in judgments:
                out.remove('unification')

        if not with_inadmissible:
            # remove inadmissible because it is a formal reason and not that interesting semantically.
            # Facts are formulated/summarized in a way to justify the decision of inadmissibility
            # hard to know solely because of the facts (formal reasons, not visible from the facts)
            if 'inadmissible' in judgments:
                out.remove('inadmissible')

        # remove all labels which are complex combinations (reason: different questions => model cannot know which one to pick)
        if make_single_label:
            # contrary judgments point to multiple questions which is too complicated
            if 'dismissal' in out and 'approval' in out:
                return np.nan
            # if we have inadmissible and another one, we just remove inadmissible
            if 'inadmissible' in out and len(out) > 1:
                out.remove('inadmissible')
            if len(out) > 1:
                message = f"By now we should only have one label. But instead we still have the labels {out}"
                raise ValueError(message)
            elif len(out) == 1:
                return out.pop()  # just return the first label because we only have one left
            else:
                return np.nan

        return list(out)

    df.judgments = df.judgments.apply(clean)
    return df

# according to BFS: https://en.wikipedia.org/wiki/Subdivisions_of_Switzerland
regions = {
    "Eastern_Switzerland": ["SG", "TG", "AI", "AR", "GL", "SH", "GR"],
    "Zürich": ["ZH"],
    "Central_Switzerland": ["UR", "SZ", "OW", "NW", "LU", "ZG"],
    "Northwestern_Switzerland": ["BS", "BL", "AG"],
    "Espace_Mittelland": ["BE", "SO", "FR", "NE", "JU"],
    "Région lémanique": ["GE", "VD", "VS"],
    "Ticino": ["TI"],
    "Federation": ["CH"],  # this is a hack to map CH to a region too
}


def get_region(canton: str):
    if canton is None:
        return None
    for region, cantons in regions.items():
        if canton in cantons:
            return region
    raise ValueError(f"Please provide a valid canton name. Could not find {canton} in {regions}")


legal_areas = {
    "public_law": ['CH_BGer_001', 'CH_BGer_002'],
    "civil_law": ['CH_BGer_004', 'CH_BGer_005'],
    "penal_law": ['CH_BGer_006', 'CH_BGer_011', 'CH_BGer_013'],
    "social_law": ['CH_BGer_008', 'CH_BGer_009'],
    "insurance_law": ['CH_BGer_016'],
    "other": ['CH_BGer_010', 'CH_BGer_012', 'CH_BGer_014', 'CH_BGer_015', 'CH_BGer_999'],
}


def get_legal_area(chamber: str):
    if chamber is None:
        return None
    if not chamber.startswith('CH_BGer_'):
        raise ValueError("So far this method is only implemented for the Federal Supreme Court")

    for legal_area, chambers in legal_areas.items():
        if chamber in chambers:
            return legal_area
    raise ValueError(f"Please provide a valid chamber name. Could not find {chamber} in {legal_areas}")