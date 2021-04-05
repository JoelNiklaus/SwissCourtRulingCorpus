import configparser
import json
import re
import importlib.util
from pathlib import Path
from typing import Optional

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
    "title",
    "judges",
    "parties",
    "topic",
    "situation",
    "considerations",
    "rulings",
    "footer"
]


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

        self.functions = {}
        self.load_functions(config, 'cleaning_functions')
        self.load_functions(config, 'section_splitting_functions')
        self.load_cleaning_regexes(config)

    def load_functions(self, config, type):
        """loads the cleaning functions used for html files"""
        function_file = ROOT_DIR / config['files'][type]  # mainly used for html courts
        spec = importlib.util.spec_from_file_location("cleaning_functions", function_file)
        self.functions[type] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.functions[type])
        self.logger.debug(self.functions[type])

    def load_cleaning_regexes(self, config):
        """loads the cleaning regexes used for pdf files"""
        regex_file = ROOT_DIR / config['files']['cleaning_regexes']  # mainly used for pdf spiders
        with open(regex_file) as f:
            self.cleaning_regexes = json.load(f)
        self.logger.debug(self.cleaning_regexes)

    def clean(self):
        """cleans all the raw court rulings with the defined regexes (for pdfs) and functions (for htmls)"""
        self.logger.info("Starting to clean raw court rulings")
        raw_list = [Path(spider).stem for spider in glob.glob(f"{str(self.raw_subdir)}/*")]
        self.logger.info(f"Found {len(raw_list)} spiders in total")

        clean_list = [Path(spider).stem for spider in glob.glob(f"{str(self.clean_subdir)}/*")]
        self.logger.info(f"Found {len(clean_list)} spiders already cleaned: {clean_list}")

        not_yet_cleaned_spiders = set(raw_list) - set(clean_list)
        spiders_to_clean = [spider for spider in not_yet_cleaned_spiders if spider[0] != '_']  # exclude aggregations
        self.logger.info(f"Still {len(spiders_to_clean)} spiders(s) remaining to clean: {spiders_to_clean}")

        for spider in spiders_to_clean:
            self.clean_spider(spider)

        self.logger.info("Finished cleaning raw court rulings")

    def clean_spider(self, spider):
        """Cleans one spider csv file"""
        # dtype_dict = {key: 'string' for key in court_keys}  # create dtype_dict from court keys
        # df = pd.read_csv(self.raw_subdir / (spider + '.csv'), dtype=dtype_dict)  # read df of spider
        self.logger.info(f"Started cleaning {spider}")
        df = pd.read_parquet(self.raw_subdir / (spider + '.parquet'))  # read df of spider
        self.logger.info(f"Read into memory {len(df.index)} decisions")

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        self.logger.info('Standardized date')

        for section in sections:  # add empty section columns
            df[section] = ""

        ddf = dd.from_pandas(df, chunksize=1000)  # according to docs you should aim for a partition size of 100MB
        ddf = ddf.apply(self.clean_df_row, axis='columns', meta=ddf)  # apply cleaning function to each row

        ddf = ddf.dropna(subset=['text'])  # drop rows which have no text
        ddf['text'] = ddf['text'].astype(str)  # set type to string again so it can be saved to parquet

        # keep raw output in case we need it later
        # ddf = ddf.drop(['html_raw', 'pdf_raw', 'html_clean', 'pdf_clean'], axis='columns')  # remove old columns
        with ProgressBar():
            df = ddf.compute(scheduler='processes')
        self.logger.info('Cleaned html and pdf content')

        df.to_csv(self.clean_subdir / (spider + '.csv'), index=False)  # save cleaned df of spider
        df.to_parquet(self.clean_subdir / (spider + '.parquet'), index=False)  # needs pyarrow installed
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
           html_clean = self.clean_html(spider, html_raw, namespace)

        pdf_raw = series['pdf_raw']
        if pd.notna(pdf_raw) and pdf_raw not in [None, '']:
           pdf_clean = self.clean_pdf(spider, pdf_raw, namespace)

        # Combine columns into one easy to use text column (prioritize html content if both exist)
        series['text'] = np.where(html_clean != '', html_clean, pdf_clean)
        if series['text'] == '':
           series['text'] = np.nan  # set to nan, so that it can be removed afterwards
           self.logger.warning(f"No raw text available for court decision {series['file_number']}")

        sections = self.split_sections_with_functions(spider, html_raw, namespace)
        if sections:
            series = series.replace(sections)

        return series

    def clean_pdf(self, spider: str, text: str, namespace: dict) -> str:
        """Cleans first the text first with court specific regexes and then with general ones"""
        cleaned_text = self.clean_with_regexes(spider, text, namespace)
        return clean_text(cleaned_text)

    def clean_html(self, spider: str, text: str, namespace: dict) -> str:
        """Cleans first the text first with court specific regexes and then with general ones"""
        cleaned_text = self.clean_with_functions(spider, text, namespace)
        return clean_text(cleaned_text)

    def clean_with_functions(self, spider: str, text: str, namespace: dict) -> str:
        """Cleans html documents with cleaning functions"""
        # Parses the html string with bs4 and returns the body content
        soup = bs4.BeautifulSoup(text, "html.parser").find('body')
        assert soup
        if not hasattr(self.functions['cleaning_functions'], spider):
            self.logger.debug(f"There are no special functions for spider {spider}. Just performing default cleaning.")
        else:
            cleaning_function = getattr(self.functions['cleaning_functions'],
                                        spider)  # retrieve cleaning function by spider
            soup = cleaning_function(soup, namespace)  # invoke cleaning function with soup and namespace

        # we cannot just remove tables because sometimes the content of the entire court decision is inside a table (GL_Omni)
        # for table in soup.find_all("table"):
        #    table.decompose()  # remove all tables from content because they are not useful in raw text
        return soup.get_text()

    def split_sections_with_functions(self, spider: str, text: str, namespace: dict) -> Optional[dict]:
        """Splits sections with section splitting functions"""
        # Parses the html string with bs4 and returns the body content
        soup = bs4.BeautifulSoup(text, "html.parser").find('body')
        assert soup
        if not hasattr(self.functions['section_splitting_functions'], spider):
            self.logger.debug(
                f"There are no special functions for spider {spider}. Not performing any section splitting.")
            return None
        else:
            section_splitting_functions = getattr(self.functions['section_splitting_functions'],
                                                  spider)  # retrieve cleaning function by spider
            try:
                return section_splitting_functions(soup, namespace)  # invoke cleaning function with soup and namespace
            except ValueError:
                return None  # just ignore the error for now. It would need much more rules to prevent this.

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
