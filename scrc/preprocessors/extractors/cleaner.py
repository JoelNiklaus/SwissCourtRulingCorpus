import configparser
import json
import re
from typing import List, Optional, Any

import bs4
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from sqlalchemy.engine.base import Engine
from sqlalchemy.sql.expression import insert

from root import ROOT_DIR
from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import clean_text, get_config
from scrc.utils.sql_select_utils import join_decision_and_language_on_parameter, join_decision_on_parameter, join_tables_on_decision, where_decisionid_in_list, where_string_spider


class Cleaner(AbstractExtractor):
    """
    Cleans the different courts from unnecessary strings to improve further processing.

    IMPORTANT: Make sure you have enough free RAM to run this (especially to clean the bigger spiders like BGer)

    For the biggest courts by size, special cleaning regexes (for pdfs)
    and cleaning functions (for htmls) have been developed for cleaner data.
    The regexes and functions have been tested on the basis of 3 random court decisions per court.

    Additional information to specific courts:
        - CH_BVGer:
            - sometimes contains spaces inside words: e.g. U r t e i l or B u n d e s v e r w a l t u n g s g e r i c h t
                => hard to fix because one does not know where the next word starts
        - ZH_Verwaltungsgericht:
            - contains an interesting summary at the top. But unfortunately it is quite hard to extract
    """

    def __init__(self, config: dict):
        super().__init__(config, 
                         function_name='cleaning_functions',
                         col_name='text', 
                         col_type='text')
        self.cleaning_regexes = self.load_cleaning_regexes(config['files']['cleaning_regexes'])
        self.processed_file_path = self.progress_dir / "spiders_cleaned.txt"

        self.logger_info = {
            'start': 'Started cleaning decisions',
            'finished': 'Finished cleaning decisions',
            'start_spider': 'Started cleaning decisions for spider',
            'finish_spider': 'Finished cleaning decisions for spider',
            'saving': 'Saving chunk of cleaned decisions',
            'processing_one': 'Cleaning the decision',
            'no_functions': 'Not cleaning the decision.'
        }

    def load_cleaning_regexes(self, file_name):
        """loads the cleaning regexes used for pdf files"""
        regex_file = self.spider_specific_dir / file_name  # mainly used for pdf spiders
        with open(regex_file) as f:
            cleaning_regexes = json.load(f)
        return cleaning_regexes

    def clean(self, decision_ids: Optional[List] = None):
        """cleans all the raw court rulings with the defined regexes (for pdfs) and functions (for htmls)"""
        self.logger.info("Started cleaning raw court rulings")
        processed_file_path = self.progress_dir / "spiders_cleaned.txt"

        if self.rebuild_entire_database:
            processed_file_path.unlink()
        if decision_ids is not None:
            self.decision_ids = decision_ids
        spider_list, message = self.compute_remaining_spiders(processed_file_path)
        self.logger.info(message)

    def select_df(self, engine: str, spider: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        only_given_decision_ids_string = f" AND {where_decisionid_in_list(self.decision_ids)}" if self.decision_ids is not None else ""
        
        tables = f"file {join_decision_on_parameter('file_id', 'file.file_id')} LEFT JOIN section ON section.decision_id = decision.decision_id"
        columns = f"decision.decision_id,file.file_id,file_name,html_raw,pdf_raw,'{spider}' as spider, section_id"
        where_string = f"file.file_id IN {where_string_spider('file_id', spider)} {only_given_decision_ids_string}"
        return self.select(engine, tables, columns, where=where_string, chunksize=self.chunksize)

    def get_required_data(self, series: pd.DataFrame) -> Any:
        return series['spider']

    def check_condition_before_process(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing. 
        e.g. data is required to be present for analysis"""
        return True

    def save_data_to_database(self, df: pd.DataFrame, engine: Engine):
        df['section_type_id'] = 1 # Set the Section type to full text
        self.update(engine, df, 'section', ['section_text'], self.output_dir, index_name='section_id')

    def process_one_spider(self, engine, spider):
        """Cleans one spider csv file"""
        self.logger.info(f"Started cleaning {spider}")

        #self.start_progress(engine, spider, engine)
        dfs = self.select_df(engine, spider)  # stream dfs from the db
        for df in dfs:
            # according to docs you should aim for a partition size of 100MB
            df['section_text'] = ''
            ddf = dd.from_pandas(df, npartitions=self.num_cpus)
            # apply cleaning function to each row
            ddf = ddf.apply(self.process_one_df_row, axis='columns', meta=ddf)
            with ProgressBar():
                df = ddf.compute(scheduler='processes')
            self.save_data_to_database(df, engine)
        #    self.log_progress(self.chunksize)
        #self.log_coverage(engine, spider, lang)
        self.logger.info(f"{self.logger_info['finish_spider']} {spider}")

    def process_one_df_row(self, series):
        """Cleans one row of a raw df"""
        self.logger.debug(
            f"{self.logger_info['processing_one']} {series['file_name']}")
        namespace = series.reindex(['file_id', 'pdf_url', 'html_url']).to_dict()
        spider = self.get_required_data(series)

        html_clean, pdf_clean = '', ''

        html_raw = series['html_raw']
        if pd.notna(html_raw) and html_raw not in [None, '']:
            # Parses the html string with bs4 and returns the body content
            soup = bs4.BeautifulSoup(html_raw, "html.parser").find('body')
            assert soup
            html_clean = self.clean_html(spider, soup, namespace)

        pdf_raw = series['pdf_raw']
        if pd.notna(pdf_raw) and pdf_raw not in [None, '']:
            pdf_clean = self.clean_pdf(spider, pdf_raw, namespace)

        # Combine columns into one easy to use text column (prioritize html content if both exist)
        series['section_text'] = np.where(html_clean != '', html_clean, pdf_clean)
        if series['section_text'] == '':
            # series['text'] = np.nan  # set to nan, so that it can be removed afterwards
            self.logger.warning(f"No raw text available for court decision {series['file_id']}")

        # otherwise we get an error when we want to save it into the db
        series.section_text = str(series.section_text)

        return series

    def clean_pdf(self, spider: str, text: str, namespace: dict) -> str:
        """Cleans first the text first with court specific regexes and then with general ones"""
        cleaned_text = self.clean_with_regexes(spider, text, namespace)
        return clean_text(cleaned_text)

    def clean_html(self, spider: str, soup: bs4.BeautifulSoup, namespace: dict) -> str:
        """Cleans first the text first with court specific regexes and then with general ones"""
        cleaned_text = self.clean_with_functions(spider, soup, namespace)
        return clean_text(cleaned_text)

    def clean_with_functions(self, spider: str, soup: bs4.BeautifulSoup, namespace: dict) -> str:
        """Cleans html documents with cleaning functions"""
        if not hasattr(self.processing_functions, spider):
            self.logger.debug(f"There are no special functions for spider {spider}. Just performing default cleaning.")
        else:
            cleaning_function = getattr(self.processing_functions,
                                        spider)  # retrieve cleaning function by spider
            # invoke cleaning function with soup and namespace
            soup = cleaning_function(soup, namespace)

        # we cannot just remove tables because sometimes the content of the entire court decision is inside a table (GL_Omni)
        # for table in soup.find_all("table"):
        #    table.decompose()  # remove all tables from content because they are not useful in raw text
        return soup.get_text()

    def clean_with_regexes(self, spider: str, text: str, namespace: dict) -> str:
        """Cleans pdf documents with cleaning regexes"""
        if spider not in self.cleaning_regexes or not self.cleaning_regexes[spider]:
            self.logger.debug(f"There are no special regexes for spider {spider}. Just performing default cleaning.")
            return text
        else:
            regexes = self.cleaning_regexes[spider]
            assert regexes  # this should not be empty
            for regex in regexes:
                # these strings can be used in the patterns and will be replaced by the variable content
                # see: https://stackoverflow.com/questions/44757222/transform-string-to-f-string
                pattern = r'' + regex['pattern']
                for key in namespace.keys():
                    # IMPORTANT: regex quantifiers (e.g. {4}) could be replaced by string format. => check first
                    if key in pattern:  # only format when the key is in the pattern.
                        # add the namespace to the pattern
                        pattern.format(**namespace)
                # perform the replacement
                cleaned_text = re.sub(pattern, regex['replacement'], text)
            return cleaned_text


if __name__ == '__main__':
    config = get_config()

    cleaner = Cleaner(config)
    cleaner.start()
