
from __future__ import annotations
from http.client import OK
from lib2to3.pgen2.pgen import DFAState
from typing import TYPE_CHECKING, Union
from pathlib import Path
from unicodedata import category
from unittest import skip
import numpy as np
import pandas as pd
import re
from scrc.preprocessors.extractors.spider_specific.section_splitting_functions import prepare_section_markers

from pip import main
from scrc.enums.section import Section
from scrc.utils.decorators import timer


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
        self.merged = []
        self.columns = ['keyword', 'totalcount', 'example']
        self.dataArray = pd.DataFrame(columns=self.columns)
        self.counter = 0
        self.spider = ''
        self.foundInstance = False
        self.test = []
        self.logger_info = {
            'start': 'Started pattern extraction',
            'finished': 'Finished pattern extraction',
            'start_spider': 'Started pattern extraction for spider',
            'finish_spider': 'Finished pattern extraction for spider',
            'saving': 'Saving chunk of recognized sections',
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

    def process_one_df_row(self, series: pd.DataFrame) -> pd.DataFrame:
        """Processes one row of a raw df"""
        self.logger.debug(
            f"{self.logger_info['processing_one']} {series['file_name']}")
        namespace = series[['date', 'html_url', 'pdf_url', 'id']].to_dict()
        namespace['language'] = Language(series['language'])
        data = self.get_required_data(series)
        assert data
        self.analyse_short(self.call_processing_function(
            series["spider"], data, namespace
        ), namespace)

    def process_one_spider(self, engine: Engine, spider: str):
        self.logger.info(self.logger_info["start_spider"] + " " + spider)
        self.spider = spider
        for lang in self.languages:
            where = self.get_database_selection_string(spider, lang)
            # TODO make quick request to see if there are decisions at all: if not, skip lang so no confusing report is printed
            self.start_progress(engine, spider, lang)
            # stream dfs from the db
            dfs = self.select(engine, lang, where=where,
                              chunksize=self.chunksize)
            for df in dfs:
                df.apply(self.process_one_df_row, axis="columns")

    def analyse_regex(self, paragraphs: list):
        self.counter += 1
        newRow = [1] + [0] * len(self.variations)
        listToAppend = []
        # if paragraphs is not None:
        for item in paragraphs:
            if item is not None:
                if len(item) < 45:
                    foundInstance = self.iterate_variations(
                        item, self.dataArray)
                    if not foundInstance:
                        listToAppend.append([item] + newRow)
        dfTemp = pd.DataFrame(listToAppend, columns=self.columns)
        self.dataArray = pd.concat(
            [self.dataArray, dfTemp], ignore_index=True)
        self.dataArray = self.dataArray.infer_objects()
        if self.counter == 2000:
            self.dropRows(self.dataArray)
            self.dataArray.sort_values(
                by=['totalCount'], ascending=False).to_csv(f'{self.spider}.csv')

    def analyse_short(self, paragraphs: list, namespace: dict):
        url = namespace['pdf_url']
        if url == '':
            url = namespace['html_url']
        self.counter += 1
        listToAppend = []
        noDuplicates = list(dict.fromkeys(paragraphs))
        for item in noDuplicates:
            if item is not None and (4 < len(item) < 45):
                foundInstance = self.check_existance(item, self.dataArray)
                if not foundInstance:
                    listToAppend.append([item, 1, url])
        dfTemp = pd.DataFrame(listToAppend, columns=self.columns)
        self.dataArray = pd.concat(
            [self.dataArray, dfTemp], ignore_index=True
        )

        if self.counter % 100 == 0:
            print(self.counter)
  
        if self.counter == self.total_to_process:
            df: DataFrame = self.searchTable(self.dataArray)
            self.assignSection(df, namespace)

    def check_existance(self, item: str, dataArray: DataFrame):
        indices = np.where(
            dataArray["keyword"] == item)
        if indices[0].size > 0:
            dataArray.at[indices[0][0], 'totalcount'] += 1
            if indices[0].size > 1:
                self.mergeRows(indices[0], dataArray)
            return True
        return False
    
    def mergeRows(self, indices, dataArray: DataFrame):
        for index in indices[1:]:
            dataArray.at[indices[0],
                         'totalcount'] += dataArray.at[index, 'totalcount']
        dataArray.drop(indices[1:], inplace=True)
        dataArray.reset_index(drop=True, inplace=True)

    def apply_regex(self, paragraph: str, pattern):
        sentence = re.escape(paragraph)
        wildcard = pattern[0] + sentence + pattern[1]
        return wildcard
    
    def findSimilar(self, dataArray: DataFrame):
        indexList = np.array(dataArray['keyword'].tolist())
        skipIndex = []
        for index, keyword in enumerate(indexList):
            for idx, element in enumerate(self.variations):
                if idx not in skipIndex:
                    pattern = self.apply_regex(keyword, element)
                    instance = [i for i, item in enumerate(
                        indexList) if re.search(pattern, item) and item != keyword]
                    if instance:
                        print(index, len(indexList))
                        dataArray = self.searchTable(index ,dataArray, instance, self.variations[idx])
                        skipIndex += instance
        return dataArray
    
    def searchTable(self, dataArray: DataFrame):
        s = pd.Series(dataArray['keyword'])
        condition = True
        startingIndex = 0
        while(condition):
            sliced = s.iloc[startingIndex:]
            if len(sliced) == 0:
                return dataArray
            for index, value in sliced.items():
                startingIndex +=1
                foundInstance = False
                for idx, element in enumerate(self.variations):
                    pattern = self.apply_regex(value, element)
                    instances = s.str.match(pattern)
                    indexes = instances[instances].index
                    if len(indexes > 0):
                        foundInstance = True
                        dataArray = self.newSearch(dataArray, index, indexes, self.variations[idx])
                        s = pd.Series(dataArray['keyword'])
                if foundInstance:
                    break
        return dataArray
    
    def newSearch(self, df: DataFrame, currentIdx, indexes, currentPattern):
        if not (str(currentPattern) in df.columns):
            df[str(currentPattern)] = 0
        for index in indexes:
            amount = df.at[index, 'totalcount']      
            df.at[currentIdx, 'totalcount'] += amount
            df.at[currentIdx, str(currentPattern)] += amount
            df.drop(index, inplace=True)
        return df
            
    def iterate_variations(self, item: str, df: DataFrame):
        foundInstance = False
        indexList = df['keyword'].tolist()
        for idx, element in enumerate(self.variations):
            pattern = self.apply_regex(item, element)
            instance = [i for i, item in enumerate(
                indexList) if re.search(pattern, item)]
            if instance:
                foundInstance = True
                if len(instance) > 1:
                    self.searchTable(element,
                        df,
                        instance, indexList, idx, foundInstance)
                else:
                    df.at[instance[0], element] += 1
                    df.at[instance[0], 'totalCount'] += 1
        return foundInstance
    

    def temp(self, idx, df: DataFrame, indexes: list, currentPattern):
        if not (str(currentPattern) in df.columns):
            df[str(currentPattern)] = 0
        for index in indexes:
            if index in df.index:
                amount = df.at[index, 'totalcount']
                if idx not in df.index:
                    print("why")
                df.at[idx, 'totalcount'] += amount
                df.at[idx, str(currentPattern)] += amount
                df.drop(index, inplace=True)
        # selectedRows: DataFrame = dataArray.loc[indexes, 'totalcount']
        # subset = selectedRows.sum(axis=0, skipna=True)
        # dataArray.loc[idx, 'totalcount'] += subset
        # dataArray.loc[idx, str(currentPattern)] += subset
        
        df.to_csv(self.getPath(self.spider))
        return df
    
    def assignSection(self, df: DataFrame, namespace):
    
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
            foundAssigntment = False
            for key in section_markers:
                if re.search(section_markers[key], element):
                     dfs[key] = dfs[key].append(df.loc[index], ignore_index=True)
                     foundAssigntment = True
            if not foundAssigntment:
                dfs['unknown'] = dfs['unknown'].append(df.loc[index], ignore_index=True)
        self.dfToCsv(dfs)
             
    def dfToCsv(self, dfs):
        with pd.ExcelWriter(self.getPath(self.spider)) as writer:
          for key in dfs:
              dfs[key].sort_values(
              by=['totalcount'], ascending=False).to_excel(writer, sheet_name=str(key))

    def dropRows(self, dataArray: DataFrame):
        dataArray.drop(
            dataArray[dataArray.totalcount < 2].index, inplace=True)
        dataArray.reset_index(drop=True, inplace=True)

    def getPath(self, spider: str):
        filepath = Path(
            f'data/patterns/{spider}.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        return filepath


if __name__ == '__main__':
    config = get_config()
    section_splitter = PatternExtractor(config)
    section_splitter.start()
