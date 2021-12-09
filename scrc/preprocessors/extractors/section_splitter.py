from __future__ import annotations
import configparser
from datetime import datetime
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union

import bs4
import pandas as pd
import uuid

from scrc.enums.language import Language
from scrc.enums.section import Section
from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor
from root import ROOT_DIR
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config

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

    """log_coverage for users without database write privileges, parses results from json logs
       allows to print coverage half way through with the coverage for the parsed chunks still being valid"""
    def log_coverage_from_json(self, engine: Engine, spider: str, lang: str, batch_info: dict) -> None:
        
        if self.total_to_process == 0:
            # no entries to process
            return
        
        path = Path.joinpath(self.output_dir, os.getlogin(), spider, lang, batch_info['uuid'])
        
        # summary: all collected elements and the count of found sections, initialized to zero
        summary = { 'total_collected': 0 }
        for section in Section:
            summary[section.value] = 0
        
        # retrieve all chunks iteratively to limit memory usage
        for chunk in range(batch_info['chunknumber']):
            df_chunk = pd.read_json(str(path / f"{chunk}.json"))
            summary['total_collected'] += df_chunk.shape[0]
            for section in Section:
                summary[section.value] += int((df_chunk[section.value] != '').sum())

        if summary['total_collected'] == 0:
            self.logger.info(f"Could not find any stored log files for batch {batch_info['uuid']} in {lang}")
            return

        total_processed = self.total_to_process
        # ceil division to get the number of files (chunks) that should be present if pipeline finished
        total_chunks = -(self.total_to_process // - self.chunksize)
        # if the pipeline was interrupted, notify the user, this allows to print coverage half way through
        # with the coverage for the parsed chunks still being valid
        if total_chunks > batch_info['chunknumber']:
            self.logger.info("The pipeline was interrupted or logged intermittently, " \
                             f"the last chunk was {batch_info['uuid']} with number {str(batch_info['chunknumber'])}")
            self.logger.info(f"Coverage for the processed chunks:")
            # update total to give a correct coverage
            total_processed = summary['total_collected']
        else:
            self.logger.info(f"{self.logger_info['finish_spider']} in {lang} with the following amount recognized:")
        for section in Section:
            section_amount = summary[section.value]
            self.logger.info(
                f"{section.value.capitalize()}:\t{section_amount} / {total_processed} "
                f"({section_amount / total_processed:.2%}) "
            )

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
            # passing batch_info to store results in json readable for the coverage report if readonly user
            batchinfo = {'uuid': uuid.uuid4().hex[:8], 'chunknumber': 0}
            log_dir = Path.joinpath(self.output_dir, os.getlogin(), spider, lang, batchinfo['uuid'])
            for df in dfs:
                df = df.apply(self.process_one_df_row, axis='columns')
                filename = f"{batchinfo['chunknumber']}.json"
                self.update(engine, df, lang, [section.value for section in Section] + ['paragraphs'], log_dir, filename)
                self.log_progress(self.chunksize)
                batchinfo['chunknumber'] += 1
                
            if self._check_write_privilege(engine):
                self.log_coverage(engine, spider, lang)
            else:
                self.log_coverage_from_json(engine, spider, lang, batchinfo)

        self.logger.info(f"{self.logger_info['finish_spider']} {spider}")

    def process_one_df_row(self, series: pd.DataFrame) -> pd.DataFrame:
        """Override method to handle section data and paragraph data individually"""
        # TODO consider removing the overriding function altogether with new db
        self.logger.debug(f"{self.logger_info['processing_one']} {series['file_name']}")
        namespace = series[['date', 'html_url', 'pdf_url', 'id']].to_dict()
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
    config = get_config()
    section_splitter = SectionSplitter(config)
    section_splitter.start()
