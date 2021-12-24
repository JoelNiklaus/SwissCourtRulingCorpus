from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional
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
    """Not tested implementation"""
    
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
        key_to_id = {
            'de': 1,
            'fr': 2,
            'it': 3,
            'en': 4
        }
        series['file_id'] = pd.read_sql(f"SELECT file_id FROM file WHERE file_name = {series['file_name']}", engine.connect())["file_id"][0]
        series['language_id'] = key_to_id[series.language]
        series['chamber_id'] = pd.read_sql(f"SELECT chamber_id FROM chamber WHERE chamber_string = {series['chamber']}", engine.connect())['chamber_id'][0]
        series['topic'] = ''
        return series
        
    def save_the_file_numbers(series: pd.DataFrame) -> pd.DataFrame:
        series['decision_id'] = pd.read_sql(f"SELECT decision_id FROM decision WHERE file_id = {series['file_id']}", engine.connect())["decision_id"][0]
        series['text'] = series['file_number']
        save_to_db([['decision_id', 'text']], 'file_number')
        if ('file_number_additional' in series and series['file_number_additional'] is not None and len(series['file_number_additional'])>0):
            series['text'] = series['file_number_additional']
            save_to_db([['decision_id', 'text']], 'file_number')
        return series
    
        
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
    
def map_join(fields: List[str], map_field: str, new_map_field_name: str, table: str, group: str = 'decision_id', join_table: str = 'd', additional_join_string: str = '') -> str:
    """ Joins a table and concatenates multiple value onto one line """
    fields_string = ", ".join(fields)
    return f" LEFT JOIN (SELECT {fields_string}, array_agg({map_field}) {new_map_field_name} FROM {table} {additional_join_string} GROUP BY {group}) as {table} ON {table}.{group} = {join_table}.{group}"

def join_decision_with_everything_joined() -> str:
    """Join every table onto decision except party, judicial_person person, party_type, judicial_person_type, paragraph

    Returns:
        str: The join string
    """
    tables = "decision d"
    tables += join('citation') + ' LEFT JOIN citation_type ON citation_type.citation_type_id = citation.citation_type_id'
    tables += join('file', 'file_id')
    tables += join('section') + ' LEFT JOIN section_type ON section_type.section_type_id = section.section_type_id'
    tables += join('lower_court')
    tables += join('language', 'language_id')
    tables += join('chamber', 'chamber_id') + ' LEFT JOIN court ON court.court_id = chamber.court_id LEFT JOIN spider ON chamber.spider_id = spider.spider_id'
    tables += map_join(['decision_id', 'text as judgment_text'], 'judgment_id', 'judgments', 'judgment_map', additional_join_string=' LEFT JOIN judgment ON judgment.judgment_id = judgment_map.judgment_id')
    tables += map_join(['text as file_number_text'], 'file_number_id', 'file_numbers', 'file_number')
    
        
    return  tables

    
def join_tables_on_decision(tables: List[str]) -> str:
    """Usage SELECT <FIELDS YOU WANT, PREFIXED WITH TABLENAME> FROM f"join_tables_on_decision(['TABLE_1', 'TABLE_2'])"

    Args:
        tables (List[str]): [description]

    Returns:
        str: [description]
    """
    join_string = 'decision d'
    
    if ('citation' in join_string or 'citation_type' in join_string):
        join_string += join('citation') + ' LEFT JOIN citation_type ON citation_type.citation_type_id = citation.citation_type_id'
    
    if ('file' in join_string):
        join_string += join('file', 'file_id')
        
    if ('section' in join_string or 'section_type' in join_string):
        join_string += join('section') + ' LEFT JOIN section_type ON section_type.section_type_id = section.section_type_id'
        
    if ('lower_court' in join_string):
        join_string += join('lower_court')
        
    if ('language' in join_string):
        join_string += join('language', 'language_id')
        
    if ('chamber' in join_string or 'court' in join_string or 'spider' in join_string):
        join_string += join('chamber', 'chamber_id') + ' LEFT JOIN court ON court.court_id = chamber.court_id LEFT JOIN spider ON chamber.spider_id = spider.spider_id'
        
    if ('judgment_map' in join_string or 'judgment' in join_string):
        join_string += map_join(['decision_id', 'text as judgment_text'], 'judgment_id', 'judgments', 'judgment_map', additional_join_string=' LEFT JOIN judgment ON judgment.judgment_id = judgment_map.judgment_id')
        
    if ('file_number' in join_string):
        tables += map_join(['text as file_number_text'], 'file_number_id', 'file_numbers', 'file_number')
    
    return join_string

def select_fields_from_table(fields: List[str], table): 
    fields_strings = [f"{table}.{field}" for field in fields]
    return ", ".join(fields_strings)