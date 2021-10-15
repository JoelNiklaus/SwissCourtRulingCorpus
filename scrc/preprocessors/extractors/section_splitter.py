from __future__ import annotations
import configparser
from typing import Optional, TYPE_CHECKING, Union

import bs4
import pandas as pd

from scrc.enums.language import Language
from scrc.enums.section import Section
from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor
from root import ROOT_DIR
from scrc.utils.log_utils import get_logger

if TYPE_CHECKING:
    from sqlalchemy.engine.base import Engine


class SectionSplitter(AbstractExtractor):
    """
    Splits the raw html into sections which can later separately be used. This represents the section splitting task.
    """

    def __init__(self, config: dict):
        super().__init__(config, function_name='section_splitting_functions', col_name='')
        self.logger = get_logger(__name__)
        self.logger_info = {
            'start': 'Started section splitting',
            'finished': 'Finished section splitting',
            'start_spider': 'Started splitting sections for spider',
            'finish_spider': 'Finished splitting sections for spider',
            'saving': 'Saving chunk of recognized sections',
            'processing_one': 'Splitting sections from file',
            'no_functions': 'Not splitting into sections.'
        }
        self.processed_file_path = self.progress_dir / "spiders_section_split.txt"

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

    def add_columns(self, engine: Engine) -> None:
        """Override method to add more than one column"""
        for lang in self.languages:
            for section in Section:  # add empty section columns
                self.add_column(engine, lang, col_name=section.value, data_type='text')
            self.add_column(engine, lang, col_name='paragraphs', data_type='jsonb')

    def read_column(self, engine: Engine, spider: str, name: str, lang: str) -> pd.DataFrame:
        query = f"SELECT count({name}) FROM {lang} WHERE {self.get_database_selection_string(spider, lang)} AND {name} <> ''"
        return pd.read_sql(query, engine.connect())['count'][0]

    def log_coverage(self, engine: Engine, spider: str, lang: str):
        """Override method to get custom coverage report"""
        self.logger.info(f"{self.logger_info['finish_spider']} in {lang} with the following amount recognized:")
        for section in Section:
            section_amount = self.read_column(engine, spider, section.value, lang)
            self.logger.info(
                f"{section.value.capitalize()}:\t{section_amount} / {self.total_to_process} "
                f"({section_amount / self.total_to_process:.2%}) "
            )

    def process_one_spider(self, engine: Engine, spider: str):
        self.logger.info(self.logger_info['start_spider'] + ' ' + spider)

        for lang in self.languages:
            where = self.get_database_selection_string(spider, lang)
            self.start_progress(engine, spider, lang)
            # stream dfs from the db
            dfs = self.select(engine, lang, where=where, chunksize=self.chunksize)
            for df in dfs:
                df = df.apply(self.process_one_df_row, axis='columns')
                self.update(engine, df, lang, [section.value for section in Section] + ['paragraphs'])
                self.log_progress(self.chunksize)

            self.log_coverage(engine, spider, lang)

        self.logger.info(f"{self.logger_info['finish_spider']} {spider}")

    def process_one_df_row(self, series: pd.DataFrame) -> pd.DataFrame:
        """Override method to handle section data and paragraph data individually"""
        # TODO consider removing the overriding function altogether with new db
        self.logger.debug(f"{self.logger_info['processing_one']} {series['file_name']}")
        namespace = series[['date', 'html_url', 'id']].to_dict()
        namespace['language'] = Language(series['language'])
        data = self.get_required_data(series)
        assert data
        try:
            paragraphs_by_section = self.call_processing_function(series['spider'], data, namespace) or None
            if paragraphs_by_section:
                for section, value in paragraphs_by_section.items():
                    series[section.value] = "\t".join(value)  # TODO save as list in new db
        except TypeError as e:
            self.logger.error(f"While processing decision {series['html_url']} caught exception {e}")
        return series


if __name__ == '__main__':
    config = configparser.ConfigParser()
    # this stops working when the script is called from the src directory!
    config.read(ROOT_DIR / 'config.ini')
    section_splitter = SectionSplitter(config)
    section_splitter.start()
