
from __future__ import annotations
from typing import TYPE_CHECKING, Union
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import re
from scrc.preprocessors.extractors.spider_specific.section_splitting_functions import prepare_section_markers

from scrc.enums.section import Section
from pandas.core.frame import DataFrame

import bs4
import pandas as pd

from scrc.enums.language import Language
from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config

if TYPE_CHECKING:
    from sqlalchemy.engine.base import Engine


class PatternExtractor(AbstractExtractor):

    def __init__(self, config: dict):
        super().__init__(config, function_name="pattern_extracting_functions", col_name='')
        self.logger = get_logger(__name__)
        self.variations = [(r'^', r'\s\d\d$'),
                           (r'^', r'.$'), (r'^', r'\s$'), (r'^', r'\s.$')]
        self.columns = ['keyword', 'totalcount', 'example']
        self.df = pd.DataFrame(columns=self.columns)
        self.language = {}
        self.total = 0
        self.counter = 0
        self.spider = ''
        self.limit = 0
        self.logger_info = {
            'start': 'Started pattern extraction',
            'finished': 'Finished pattern extraction',
            'start_spider': 'Started pattern extraction for spider',
            'finish_spider': 'Finished pattern extraction for spider',
            'processing_one': 'Extracting sections from file',
            'no_functions': 'Not extracting sections.'
        }
        self.processed_file_path = self.progress_dir / "pattern_extraction.txt"

    def get_required_data(self, series: pd.DataFrame) -> Union[bs4.BeautifulSoup, str, None]:
        """Returns the data required by the processing functions"""
        html_raw = series['html_raw']
        if pd.notna(html_raw) and html_raw not in [None, '']:
            # Parses the html string with bs4 and returns the body content
            return bs4.BeautifulSoup(html_raw, "html.parser").find('body')
        pdf_raw = series['pdf_raw']
        if pd.notna(pdf_raw) and pdf_raw not in [None, '']:
            return pdf_raw
        return None

    def get_database_selection_string(self, spider: str, lang: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        return f"spider='{spider}'"

    def add_columns(self, engine: Engine):
        return None
    
    def start_progress(self, engine: Engine, spider: str, lang: str):
        self.processed_amount = 0
        self.total_to_process = self.coverage_get_total(engine, spider, lang)
        if len(sys.argv) > 1:
            if (int(sys.argv[1]) < 1000):
                self.total_to_process = 1000
            else: 
                self.total_to_process = round(int(sys.argv[1]), -3)
        self.logger.info(f"Total: {self.total_to_process}")

    def process_one_df_row(self, series: pd.DataFrame) -> pd.DataFrame:
        """Processes one row of a raw df"""
        self.logger.debug(
            f"{self.logger_info['processing_one']} {series['file_name']}")
        namespace = series[['date', 'html_url', 'pdf_url', 'id']].to_dict()
        namespace['language'] = Language(series['language'])
        data = self.get_required_data(series)
        assert data
        self.analyze_structure(self.call_processing_function(
            series["spider"], data, namespace
        ), namespace)

    def process_one_spider(self, engine: Engine, spider: str):
        self.logger.info(self.logger_info["start_spider"] + " " + spider)
        self.spider = spider
        for lang in self.languages:
            self.start_progress(engine, spider, lang)
            where = self.get_database_selection_string(spider, lang)
            dfs = self.select(engine, lang, where=where,
                              chunksize=self.chunksize)
            for df in dfs:
                if len(sys.argv) > 1:
                    if self.counter >= self.total_to_process:
                        break
                if not self.language:
                    self.total = self.total_to_process
                    self.language = {'language': Language[lang.upper()]}
                df.apply(self.process_one_df_row, axis="columns")
        if len(sys.argv) > 2:
            self.logger.info(f"Finished pattern extraction, applying regex now")
            df: DataFrame = self.apply_regex(self.df)
        self.logger.info(f"Finished regex application, assigning Sections now")
        self.assign_section(self.df, self.language)



    def analyze_structure(self, paragraphs: list, namespace: dict):
        url = namespace['pdf_url']
        if url == '':
            url = namespace['html_url']
        self.counter += 1
        listToAppend = []
        noDuplicates = list(dict.fromkeys(paragraphs))
        for item in noDuplicates:
            if item is not None and (4 < len(item) < 45):
                foundInstance = self.check_existance(item, self.df)
                if not foundInstance:
                    listToAppend.append([item, 1, url])
        dfTemp = pd.DataFrame(listToAppend, columns=self.columns)
        self.df = pd.concat(
            [self.df, dfTemp], ignore_index=True)
        if self.counter % 500 == 0:
            self.logger.info(self.get_progress_string(self.counter, self.total_to_process, "Pattern extraction"))


    def check_existance(self, item: str, dataArray: DataFrame):
        indices = np.where(
            dataArray["keyword"] == item)
        if indices[0].size > 0:
            dataArray.at[indices[0][0], 'totalcount'] += 1
            if indices[0].size > 1:
                self.merge_rows(indices[0], dataArray)
            return True
        return False
    
    def merge_rows(self, indices, df: DataFrame):
        for index in indices[1:]:
            df.at[indices[0],
                         'totalcount'] += df.at[index, 'totalcount']
        df.drop(indices[1:], inplace=True)
        df.reset_index(drop=True, inplace=True)

    def get_pattern(self, paragraph: str, pattern):
        sentence = re.escape(paragraph)
        wildcard = pattern[0] + sentence + pattern[1]
        return wildcard
    
    def apply_regex(self, df: DataFrame):
        s = pd.Series(df['keyword'])
        condition = True
        startingIndex = 0
        while(condition):
            sliced = s.iloc[startingIndex:]
            if startingIndex % 500 == 0:    
                self.logger.info(self.get_progress_string(startingIndex, s.size, "Regex application"))
            if len(sliced) == 0:
                return df
            for index, value in sliced.items():
                startingIndex +=1
                foundInstance = False
                for idx, element in enumerate(self.variations):
                    pattern = self.get_pattern(value, element)
                    instances = s.str.match(pattern)
                    indexes = instances[instances].index
                    if len(indexes > 0):
                        foundInstance = True
                        df = self.find_matches(df, index, indexes, self.variations[idx])
                        s = pd.Series(df['keyword'])
                if foundInstance:
                    break
        return df
    
    def find_matches(self, df: DataFrame, currentIdx: int, indexes: list, currentPattern):
        if not (str(currentPattern) in df.columns):
            df[str(currentPattern)] = 0
        for index in indexes:
            amount = df.at[index, 'totalcount']      
            df.at[currentIdx, 'totalcount'] += amount
            df.at[currentIdx, str(currentPattern)] += amount
            df.drop(index, inplace=True)
        return df

    
    def assign_section(self, df: DataFrame, namespace):
        all_section_markers = {
            Language.FR: {
                Section.FACTS: [r'[F,f]ait', 'droit'],
                Section.CONSIDERATIONS: [r'[C,c]onsidère', r'[C,c]onsidérant', r'droit'],
                Section.RULINGS: [r'prononce', r'motifs'],
                Section.FOOTER: [r'président']
            },
            Language.DE: {
                Section.FACTS: [r'[S,s]achverhalt', r'[T,t]atsachen', r'[E,e]insicht in'],
                Section.CONSIDERATIONS: [r'[E,e]rwägung', r"erwägt", "ergeben"],
                Section.RULINGS: [r'erk[e,a]nnt',  r'beschlossen', r'verfügt', r'beschliesst',r'entscheidet'],
                Section.FOOTER: [r'Rechtsmittelbelehrung']
            }
        }
        section_markers = prepare_section_markers(all_section_markers, namespace)
        dfs = {}
        for key in Section:
            if key == Section.HEADER:
                key = 'unknown'
            dfs[key] = pd.DataFrame([], columns=df.columns)
        df.reset_index(inplace=True)
        for index, element in enumerate(df['keyword'].tolist()):
            if index % 500 == 0:
                self.logger.info(self.get_progress_string(index, df['keyword'].size, "Section assignment"))
            foundAssigntment = False
            for key in section_markers:
                if re.search(section_markers[key], element):
                     dfs[key] = dfs[key].append(df.loc[index], ignore_index=True)
                     foundAssigntment = True
            if not foundAssigntment:
                dfs['unknown'] = dfs['unknown'].append(df.loc[index], ignore_index=True)
        self.df_to_csv(self.calculate_coverage(dfs))
        
    def get_progress_string(self, progress: int, total: int, name: str):
        return f"{name}: {progress} of {total} processed"
        
    def calculate_coverage(self, dfs):
        for key in dfs:
            if key == 'unknown':
                dfs[key] = self.drop_rows(dfs[key])
            dfs[key]['coverage'] = 0
            dfs[key]['coverage'] = dfs[key].apply(self.get_percentage, axis = 1)
            dfs[key].drop(columns=['index'], inplace=True)
        return dfs

    def get_percentage(self, s):
        return s.at['totalcount'] / self.total * 100
             
    def df_to_csv(self, dfs):
        with pd.ExcelWriter(self.get_path(self.spider)) as writer:
          for key in dfs:
              dfs[key].sort_values(
              by=['totalcount'], ascending=False).to_excel(writer, sheet_name=str(key), index=False)

    def drop_rows(self, df: DataFrame):
        df.drop(
            df[df.totalcount < 2].index, inplace=True) 
        df.reset_index(drop=True, inplace=True)
        return df

    def get_path(self, spider: str):
        filepath = Path(
            f'data/patterns/{spider}.xlsx')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        return filepath


if __name__ == '__main__':
    config = get_config()
    section_splitter = PatternExtractor(config)
    section_splitter.start()
