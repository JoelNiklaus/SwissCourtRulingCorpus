from __future__ import annotations
import configparser
import datetime
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING
from nltk.tokenize import sent_tokenize



import pandas as pd
from sqlalchemy.engine.base import Engine
from sqlalchemy.sql.expression import text
from sqlalchemy.sql.schema import MetaData, Table
from scrc.enums.judgment import Judgment
from scrc.enums.language import Language
from scrc.preprocessors.abstract_preprocessor import AbstractPreprocessor
from nltk import ngrams



from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor
from root import ROOT_DIR
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config
from scrc.utils.sql_select_utils import delete_stmt_decisions_with_df, join_decision_and_language_on_parameter, \
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
        self.processed_file_path = self.progress_dir / "judgment_pattern_extractor.txt"
        self.dict = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}}
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
    
    def get_coverage(self, engine: Engine, spider: str):
        """Splits the data into their respective parts and saves them to the table"""
    
    def get_coverage(self, engine: Engine, spider: str):
        """Splits the data into their respective parts and saves them to the table"""
        
    def process_one_spider(self, engine: Engine, spider: str):
        self.logger.info(self.logger_info["start_spider"] + " " + spider)
        dfs = self.select_df(self.get_engine(self.db_scrc), spider)  # Get the data needed for the extraction
        # TODO make quick request to see if there are decisions at all: if not, skip lang so no confusing report is printed
        # self.start_progress(engine, spider)
        # stream dfs from the db
        # dfs = self.select(engine, lang, where=where, chunksize=self.chunksize)
        for df in dfs:  # For each chunk in the data: apply the extraction function and save the result
            df = df.apply(self.process_one_df_row, axis="columns")
            self.df_to_csv(spider)
        self.logger.info(f"{self.logger_info['finish_spider']} {spider}")
    
    def process_one_df_row(self, series: pd.DataFrame) -> pd.DataFrame:
        """Processes one row of a raw df"""
        namespace = dict()
        # Add the data to the namespace object which is passed to the extraction function
        namespace['html_url'] = series.get('html_url')
        namespace['pdf_url'] = series.get('pdf_url')
        namespace['date'] = series.get('date')
        namespace['language'] = Language(series['language'])
        namespace['id'] = series['decision_id']
        if 'court_string' in series:
            namespace['court'] = series.get('court_string')
        data = self.get_required_data(series)
        if data: 
            self.sentencize(data)
              
    
    def df_to_csv(self, spider):
        # todo: section assignment
        with pd.ExcelWriter(self.get_path(spider)) as writer:
            for key in self.dict:
                dict_list = list(self.dict[key].items())
                dict_list.sort(key = lambda x:x[1]['totalcount'], reverse = True)
                df = pd.DataFrame(dict_list)
                df.to_excel(writer, sheet_name=str(key),index=False)
                
    def get_path(self, spider: str):
        filepath = Path(f'data/judgment_patterns/{spider}.xlsx')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        return filepath
            
        
    
    def add_combinations(self, sentence):
        # add url
        for i in range(1, 7):
            gram_list = self.create_ngrams(i, sentence)
            for gram in gram_list:
                if gram in self.dict[i]:
                    self.dict[i][gram]['totalcount'] += 1
                else:
                    self.dict[i][gram] = {'totalcount': 1}
            
    def create_ngrams(self, n, sentence):
        n_gram = ngrams(sentence, n)
        return [gram for gram in n_gram]
    
    def sentencize(self, data):
        sentence_list = sent_tokenize(data)
        for sentence in sentence_list:
            self.add_combinations(sentence.split())
        

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
        """Do nothing"""

    def check_condition_before_process(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing.
        e.g. data is required to be present for analysis"""
        return bool(data)


if __name__ == '__main__':
    config = get_config()
    judgment_extractor = JudgmentExtractor(config)
    judgment_extractor.start()
