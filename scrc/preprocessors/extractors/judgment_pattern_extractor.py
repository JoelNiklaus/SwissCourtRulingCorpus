from __future__ import annotations
from ast import Set
from pathlib import Path
import re
import sys
from typing import Any, TYPE_CHECKING
from nltk.tokenize import sent_tokenize
from scrc.utils.main_utils import clean_text, int_to_roman



import pandas as pd
from sqlalchemy.engine.base import Engine
from scrc.enums.judgment import Judgment
from scrc.enums.language import Language
from nltk import ngrams
from scrc.enums.section import Section


from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config
from scrc.utils.sql_select_utils import get_judgment_query, get_total_decisions, get_total_judgments, join_decision_and_language_on_parameter, \
    join_file_on_decision, where_decisionid_in_list, where_string_spider
    
from scrc.preprocessors.extractors.spider_specific.judgment_extracting_functions import get_nth_ruling, prepare_judgment_markers, search_rulings

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame


class JudgmentExtractor(AbstractExtractor):
    """
    Extracts the pattern of ruling section indicators for a given court. Only outputs the coverage if a command line argument it given.
    """

    def __init__(self, config: dict):
        super().__init__(config, function_name='judgment_extracting_functions', col_name='judgments')
        self.logger = get_logger(__name__)
        self.processed_file_path = self.progress_dir / "judgment_pattern_extractor.txt"
        self.dict = {}
        self.language = ""
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
        ruling_id = Section.RULINGS.value
        with self.get_engine(self.db_scrc).connect() as conn:
            total_judgments = conn.execute(get_total_judgments(spider, ruling_id)).fetchone()
            coverage_result = conn.execute(get_judgment_query(spider)).fetchone()
            total_cases = conn.execute(get_total_decisions(spider)).fetchone()
            coverage =  round(coverage_result[0] / total_judgments[0]  * 100, 2)
            self.logger.info(f'{spider}:\n Found judgment outcome for {coverage}% of the rulings.\nJudgments found: {coverage_result[0]}.\nTotal non-empty rulings: {total_judgments[0]}\nTotal cases: {total_cases[0]}')
    

        
    def start_spider_loop(self, spider_list: Set, engine: Engine):
        for spider in spider_list:
            if len(sys.argv) > 1:
                self.get_coverage(spider)
                self.mark_as_processed(self.processed_file_path, spider)
            else:
                self.init_dict()
                self.process_one_spider(engine, spider)
                self.mark_as_processed(self.processed_file_path, spider)
            
    def drop_rows(self, df: DataFrame):
        if 'totalcount' in df.columns:
            df.drop(
                df[df.totalcount < 4].index, inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

        
    def process_one_spider(self, engine: Engine, spider: str):
        self.logger.info(self.logger_info["start_spider"] + " " + spider)
        dfs = self.select_df(self.get_engine(self.db_scrc), spider)  # Get the data needed for the extraction
        for idx, df in enumerate(dfs):  
            df = df.apply(self.process_one_df_row, axis="columns")
            self.logger.info(f"{idx + 1} df processed")
        assigned_lists = self.assign_section()
        self.df_to_csv(spider, assigned_lists)
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
            self.sentencize(data, namespace)
              
    
    def df_to_csv(self, spider, assigned_lists):
        for language in assigned_lists:
            with pd.ExcelWriter(self.get_path(spider, language)) as writer:
                for key in assigned_lists[language]:
                    list_to_df = [{'value': element[0], 'totalcount': element[1]['totalcount'], 'url': element[1]['url']} for element in assigned_lists[language][key]]
                    df = self.drop_rows(pd.DataFrame(list_to_df))
                    df.to_excel(writer, sheet_name=str(key),index=False)
                
    def get_path(self, spider: str, lang):
        filepath = Path(f'data/judgment_patterns/{spider}/{spider}_{lang}_judg.xlsx')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        return filepath
            
    def sort_dict(self):
        final_dict = {}
        for key in self.dict:
            if bool(self.dict[key]):
                dict_list = list(self.dict[key].items())
                dict_list.sort(key = lambda x:x[1]['totalcount'], reverse = True)
                final_dict[key] = dict_list
        return final_dict
    
    def init_dict(self):
        self.dict = {Language.DE: {}, Language.FR: {}, Language.IT: {}, Language.EN: {}, Language.UK: {}}
    
    def add_combinations(self, sentence, url, namespace):
        for i in range(1, 10):
            gram_list = self.create_ngrams(i, sentence)
            for gram in gram_list:
                if gram in self.dict[namespace['language']]:
                    self.dict[namespace['language']][gram]['totalcount'] += 1
                else:
                    self.dict[namespace['language']][gram] = {'totalcount': 1, 'url': url}
            
    def create_ngrams(self, n, sentence):
        n_gram = ngrams(sentence, n)
        return [gram for gram in n_gram]
    
    def sentencize(self, data, namespace):
        data = self.numbered_ruling(data)
        url = namespace['html_url']
        if url == '':
            url = namespace['pdf_url']
        sentence_list = sent_tokenize(data)
        for sentence in sentence_list:
            self.add_combinations(sentence.split(), url, namespace)
      
    def numbered_ruling(self, data):
        ruling = clean_text(data)
        result = search_rulings(ruling, str(1), str(2))
        if not result:
            result = search_rulings(ruling, int_to_roman(1), int_to_roman(2))
        if result:
            return result.group(1)
        return ruling

    def get_required_data(self, series: DataFrame) -> Any:
        """Returns the data required by the processing functions"""
        return series['section_text']

    def select_df(self, engine: str, spider: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        only_given_decision_ids_string = f" AND {where_decisionid_in_list(self.decision_ids)}" if self.decision_ids is not None else ""
        ruling_id = Section.RULINGS.value
        return self.select(engine,
                           f"section {join_decision_and_language_on_parameter('decision_id', 'section.decision_id')} {join_file_on_decision()}",
                           f"section.decision_id, section_text, '{spider}' as spider, iso_code as language, html_url",
                           where=f"section.section_type_id = {ruling_id} AND section.decision_id IN {where_string_spider('decision_id', spider)} {only_given_decision_ids_string}",
                           chunksize=self.chunksize)

    def save_data_to_database(self, df: pd.DataFrame, engine: Engine):
        """Do nothing"""
        
    def assign_section(self):
        sorted_lists = self.sort_dict()
        assigned_dict = {}
        for key in sorted_lists:
            assigned_dict[key] = {}         
            for element in sorted_lists[key]:
                type = f"{len(element[0])}_gram"
                if(type not in assigned_dict[key]):
                    assigned_dict[key][type] = []
                string = ' '.join(element[0]) 
                assigned_dict[key][type].append((element))
        return assigned_dict
        
        
 

    def check_condition_before_process(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing.
        e.g. data is required to be present for analysis"""
        return bool(data)


if __name__ == '__main__':
    config = get_config()
    judgment_extractor = JudgmentExtractor(config)
    judgment_extractor.start()
