import configparser
import json
import re
from typing import Optional, Any

import bs4
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from root import ROOT_DIR
from scrc.preprocessing.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import clean_text


class Cleaner(DatasetConstructorComponent):
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
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.cleaning_functions = self.load_functions(config, 'cleaning_functions')
        self.logger.debug(self.cleaning_functions)

        self.cleaning_regexes = self.load_cleaning_regexes(config['files']['cleaning_regexes'])
        self.logger.debug(self.cleaning_regexes)

    def load_cleaning_regexes(self, file_name):
        """loads the cleaning regexes used for pdf files"""
        regex_file = self.spider_specific_dir / file_name  # mainly used for pdf spiders
        with open(regex_file) as f:
            cleaning_regexes = json.load(f)
        return cleaning_regexes

    def clean(self):
        """cleans all the raw court rulings with the defined regexes (for pdfs) and functions (for htmls)"""
        self.logger.info("Started cleaning raw court rulings")

        processed_file_path = self.data_dir / "spiders_cleaned.txt"
        spider_list, message = self.compute_remaining_spiders(processed_file_path)
        self.logger.info(message)

        engine = self.get_engine(self.db_scrc)
        for lang in self.languages:
            # add new columns to db
            self.add_column(engine, lang, col_name='text', data_type='text')

        # clean raw texts
        for spider in spider_list:
            self.clean_spider(engine, spider)
            self.mark_as_processed(processed_file_path, spider)

        self.logger.info("Finished cleaning raw court rulings")

    def clean_spider(self, engine, spider):
        """Cleans one spider csv file"""
        self.logger.info(f"Started cleaning {spider}")

        for lang in self.languages:
            dfs = self.select(engine, lang, where=f"spider='{spider}'")  # stream dfs from the db
            for df in dfs:
                # according to docs you should aim for a partition size of 100MB
                ddf = dd.from_pandas(df, npartitions=self.num_cpus)
                ddf = ddf.apply(self.clean_df_row, axis='columns', meta=ddf)  # apply cleaning function to each row
                with ProgressBar():
                    df = ddf.compute(scheduler='processes')

                columns = ['text']
                self.logger.info("Saving cleaned text to db")
                self.update(engine, df, lang, columns)

        self.logger.info(f"Finished cleaning {spider}")

    def clean_df_row(self, series):
        """Cleans one row of a raw df"""
        self.logger.debug(f"Cleaning court decision {series['file_name']}")
        namespace = series[
            ['file_number', 'file_number_additional', 'date', 'language', 'pdf_url', 'html_url']].to_dict()
        spider = series['spider']

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
        series['text'] = np.where(html_clean != '', html_clean, pdf_clean)
        if series['text'] == '':
            # series['text'] = np.nan  # set to nan, so that it can be removed afterwards
            self.logger.warning(f"No raw text available for court decision {series['file_number']}")

        series.text = str(series.text)  # otherwise we get an error when we want to save it into the db

        return series

    def clean_pdf(self, spider: str, text: str, namespace: dict) -> str:
        """Cleans first the text first with court specific regexes and then with general ones"""
        cleaned_text = self.clean_with_regexes(spider, text, namespace)
        return clean_text(cleaned_text)

    def clean_html(self, spider: str, soup: Any, namespace: dict) -> str:
        """Cleans first the text first with court specific regexes and then with general ones"""
        cleaned_text = self.clean_with_functions(spider, soup, namespace)
        return clean_text(cleaned_text)

    def clean_with_functions(self, spider: str, soup: Any, namespace: dict) -> str:
        """Cleans html documents with cleaning functions"""
        if not hasattr(self.cleaning_functions, spider):
            self.logger.debug(f"There are no special functions for spider {spider}. Just performing default cleaning.")
        else:
            cleaning_function = getattr(self.cleaning_functions,
                                        spider)  # retrieve cleaning function by spider
            soup = cleaning_function(soup, namespace)  # invoke cleaning function with soup and namespace

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
                        pattern.format(**namespace)  # add the namespace to the pattern
                cleaned_text = re.sub(pattern, regex['replacement'], text)  # perform the replacement
            return cleaned_text


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    cleaner = Cleaner(config)
    cleaner.clean()
