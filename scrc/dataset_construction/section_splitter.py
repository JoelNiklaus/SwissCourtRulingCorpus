import configparser
from scrc.utils.abstract_extractor import AbstractExtractor
from typing import Optional, Any

import bs4
import pandas as pd

from root import ROOT_DIR
from scrc.utils.log_utils import get_logger

sections = [
    "header",
    # "title",
    # "judges",
    # "parties",
    # "topic",
    "facts",
    "considerations",
    "rulings",
    "footer"
]


class SectionSplitter(AbstractExtractor):
    """
    Splits the raw html into sections which can later separately be used
    """

    def __init__(self, config: dict):
        super().__init__(config, function_name='section_splitting_functions', col_name='')


    def getRequiredData(self, series) -> Any:
        html_raw = series['html_raw']
        if pd.notna(html_raw) and html_raw not in [None, '']:
            # Parses the html string with bs4 and returns the body content
            return bs4.BeautifulSoup(html_raw, "html.parser").find('body')
        return None

    def getDatabaseSelectionString(self, spider, lang) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        return f"spider='{spider}'"

    def addColumns(self, engine):
        """Override method to add more than one column"""
        for lang in self.languages:
            for section in sections:  # add empty section columns
                self.add_column(engine, lang, col_name=section, data_type='text')
            self.add_column(engine, lang, col_name='paragraphs', data_type='jsonb')

    def readColumn(self, engine, name, lang):
                query = f"SELECT count({name}) FROM {lang} WHERE {self.getDatabaseSelectionString()} AND {name} <> ''"
                return pd.read_sql(query, engine.connect())['count'][0]
    
    def logCoverage(self, engine, spider, lang):
        """Override method to get custom coverage report"""
        self.logger.info(f"{self.loggerInfo['finish_spider']} in {lang} with the following amount recognized:")
        for section in sections:
            section_amount = self.readColumn(engine, section, lang)
            self.logger.info(
                f"{section.capitalize()}:\t{section_amount} / {self.totalToProcess} ()"
                f"({section_amount / self.totalToProcess:.2%}) "
            )

    def processOneDFRow(self, series):
        """Override method to handle section data and paragraph data individually"""
        self.logger.debug(f"{self.loggerInfo['processing_one']} {series['file_name']}")
        namespace = series[['date', 'language', 'html_url', 'id']].to_dict()
        data = self.getRequiredData(series)
        assert data
        series[self.col_name] = self.callProcessingFunction(series['spider'], data, namespace)
        try:
            section_data, paragraph_data = self.split_sections_with_functions(series['spider'], data, namespace)
            if section_data:
                for key, value in section_data.items():
                    series[key] = value
            if paragraph_data:
                series['paragraphs'] = paragraph_data
        except TypeError as e:
            self.logger.error(f"While processing decision {series['html_url']} caught exception {e}")
        return series

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    section_splitter = SectionSplitter(config)
    section_splitter.processed_file_path = section_splitter.data_dir / "spiders_section_split.txt"
    section_splitter.loggerInfo = {
        'start': 'Started section splitting', 
        'finished': 'Finished section splitting', 
        'start_spider': 'Started splitting sections for spider', 
        'finish_spider': 'Finished splitting sections for spider', 
        'saving': 'Saving chunk of recognized sections',
        'processing_one': 'Splitting sections from file',
        'no_functions': 'Not splitting into sections.'
        }
    section_splitter.start()
