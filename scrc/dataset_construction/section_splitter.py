import configparser
import json
import re
import importlib.util
from pathlib import Path
from typing import Optional, Any

import bs4
import glob
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from root import ROOT_DIR
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import clean_text

sections = [
    "header",
    # "title",
    # "judges",
    # "parties",
    # "topic",
    "situation",
    "considerations",
    "rulings",
    "footer"
]


class SectionSplitter(DatasetConstructorComponent):
    """
    Splits the raw html into sections which can later separately be used
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.section_splitting_functions = self.load_functions(config, 'section_splitting_functions')
        self.logger.debug(self.section_splitting_functions)

    def split_sections(self):
        """splits sections of all the raw court rulings with the defined functions"""
        self.logger.info("Started section-splitting raw court rulings")

        processed_file_path = self.data_dir / "spiders_section_split.txt"
        spider_list, message = self.compute_remaining_spiders(processed_file_path)
        self.logger.info(message)

        engine = self.get_engine(self.db_scrc)
        for lang in self.languages:
            # add new columns to db
            for section in sections:  # add empty section columns
                self.add_column(engine, lang, col_name=section, data_type='text')
            self.add_column(engine, lang, col_name='paragraphs', data_type='jsonb')

        # clean raw texts
        for spider in spider_list:
            if hasattr(self.section_splitting_functions, spider):
                self.section_split_spider(engine, spider)
            else:
                self.logger.debug(f"There are no special functions for spider {spider}. "
                                  f"Not performing any section splitting.")
            self.mark_as_processed(processed_file_path, spider)

        self.logger.info("Finished section-splitting raw court rulings")

    def section_split_spider(self, engine, spider):
        """Splits sections for one spider csv file"""
        self.logger.info(f"Started section-splitting {spider}")

        for lang in self.languages:
            dfs = self.select(engine, lang, where=f"spider='{spider}'")  # stream dfs from the db
            for df in dfs:
                # according to docs you should aim for a partition size of 100MB
                # ddf = dd.from_pandas(df, npartitions=self.num_cpus)
                # apply section splitting function to each row
                # ddf = ddf.apply(self.section_split_df_row, axis='columns', meta=ddf)
                # with ProgressBar():
                #    df = ddf.compute(scheduler='processes')
                df = df.apply(self.section_split_df_row, axis='columns')

                self.logger.info("Saving cleaned text and split sections to db")
                self.update(engine, df, lang, sections)

        self.logger.info(f"Finished section-splitting {spider}")

    def section_split_df_row(self, series):
        """Cleans one row of a raw df"""
        self.logger.debug(f"Section-splitting court decision {series['file_name']}")
        namespace = series[['date', 'language', 'html_url']].to_dict()
        spider = series['spider']

        html_raw = series['html_raw']
        if pd.notna(html_raw) and html_raw not in [None, '']:
            # Parses the html string with bs4 and returns the body content
            soup = bs4.BeautifulSoup(html_raw, "html.parser").find('body')
            assert soup

            section_data, paragraph_data = self.split_sections_with_functions(spider, soup, namespace)
            if section_data:
                for key, value in section_data.items():
                    series[key] = value
            if paragraph_data:
                series['paragraphs'] = paragraph_data

        return series

    def split_sections_with_functions(self, spider: str, soup: Any, namespace: dict) -> Optional:
        """Splits sections with section splitting functions"""
        # retrieve cleaning function by spider
        section_splitting_functions = getattr(self.section_splitting_functions, spider)
        try:
            return section_splitting_functions(soup, namespace)  # invoke cleaning function with soup and namespace
        except ValueError as e:
            self.logger.warning(e)
            return None, None  # just ignore the error for now. It would need much more rules to prevent this.


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    section_splitter = SectionSplitter(config)
    section_splitter.split_sections()
