import configparser
import json
import re
import unicodedata
import importlib.util
from pathlib import Path

import bs4
from pandarallel import pandarallel
import glob
import numpy as np
import pandas as pd

from root import ROOT_DIR
from scrc.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import court_keys

logger = get_logger(__name__)


class Cleaner(DatasetConstructorComponent):
    """
    Cleans the different courts from unnecessary strings to improve further processing.

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

        self.load_cleaning_regexes(config)
        self.load_cleaning_functions(config)

        # unfortunately, the progress bar does not work for parallel_apply: https://github.com/nalepae/pandarallel/issues/26
        pandarallel.initialize()  # for parallel pandas processing: https://github.com/nalepae/pandarallel

    def load_cleaning_functions(self, config):
        """loads the cleaning functions used for html files"""
        function_file = ROOT_DIR / config['files']['functions']  # mainly used for html courts
        spec = importlib.util.spec_from_file_location("cleaning_functions", function_file)
        self.cleaning_functions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cleaning_functions)
        print(self.cleaning_functions)

    def load_cleaning_regexes(self, config):
        """loads the cleaning regexes used for pdf files"""
        regex_file = ROOT_DIR / config['files']['regexes']  # mainly used for pdf spiders
        with open(regex_file) as f:
            self.spiders_regexes = json.load(f)
        print(self.spiders_regexes)

    def clean(self):
        """cleans all the raw court rulings with the defined regexes (for pdfs) and functions (for htmls)"""
        logger.info("Starting to clean raw court rulings")
        raw_csv_list = [Path(spider).stem for spider in glob.glob(f"{str(self.raw_csv_subdir)}/*")]
        clean_csv_list = [Path(spider).stem for spider in glob.glob(f"{str(self.clean_csv_subdir)}/*")]
        not_yet_cleaned_spiders = set(raw_csv_list) - set(clean_csv_list)
        spiders_to_clean = [court for court in not_yet_cleaned_spiders if court[0] != '_']  # exclude aggregations
        logger.info(f"Still {len(spiders_to_clean)} court(s) remaining to clean: {spiders_to_clean}")

        for spider in spiders_to_clean:
            self.clean_spider(spider)

        logger.info("Finished cleaning raw court rulings")

    def clean_spider(self, spider):
        """Cleans one spider csv file"""
        logger.info(f"Started cleaning {spider}")
        dtype_dict = {key: 'string' for key in court_keys}  # create dtype_dict from court keys
        df = pd.read_csv(self.raw_csv_subdir / (spider + '.csv'), dtype=dtype_dict)  # read df of spder

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        logger.info('Standardized date')

        # add empty columns
        df['html_clean'], df['pdf_clean'] = '', ''
        df = df.parallel_apply(self.clean_df_row, axis='columns')  # apply cleaning function to each row
        logger.info('Cleaned html and pdf content')

        # Combine columns into one easy to use text column (prioritize html content if both exist)
        df['text'] = np.where(df['html_clean'] != '', df['html_clean'], df['pdf_clean'])
        df = df.drop(['html_raw', 'pdf_raw', 'html_clean', 'pdf_clean'], axis='columns')  # remove old columns
        logger.info('Combined columns into one easy to use text column')

        df.to_csv(self.clean_csv_subdir / (spider + '.csv'), index=False)  # save cleaned df of spider
        logger.info(f"Finished cleaning {spider}")

    def clean_df_row(self, series):
        """Cleans one row of a raw df"""
        logger.debug(f"Cleaning court decision {series['file_name']}")
        namespace = series[['file_number', 'file_number_additional', 'date', 'language']].to_dict()
        spider = series['spider']

        html_raw = series['html_raw']
        if pd.notna(html_raw):
            series['html_clean'] = self.clean_html(spider, html_raw, namespace)

        pdf_raw = series['pdf_raw']
        if pd.notna(pdf_raw):
            series['pdf_clean'] = self.clean_pdf(spider, pdf_raw, namespace)

        return series

    def clean_pdf(self, spider: str, text: str, namespace: dict) -> str:
        """Cleans first the text first with court specific regexes and then with general ones"""
        cleaned_text = self.clean_with_regexes(spider, text, namespace)
        return self.clean_generally(cleaned_text)

    def clean_html(self, spider: str, text: str, namespace: dict) -> str:
        """Cleans first the text first with court specific regexes and then with general ones"""
        cleaned_text = self.clean_with_functions(spider, text, namespace)
        return self.clean_generally(cleaned_text)

    @staticmethod
    def clean_generally(text: str) -> str:
        """
        Clean text from nasty tokens
        :param text:    the text to be cleaned
        :return:
        """
        cleaned_text = text
        cleaned_text = unicodedata.normalize('NFKD', cleaned_text)  # normalize whitespace
        cleaned_text = re.sub('(\w+)-\n+(\w+)', '\1\2', cleaned_text)  # remove hyphens before new line
        cleaned_text = re.sub(r"\u00a0", ' ', cleaned_text)  # replace NBSP with normal whitespace
        cleaned_text = re.sub(r"\xa0", ' ', cleaned_text)  # replace \xa0 with normal whitespace
        cleaned_text = re.sub(r"\s+", ' ', cleaned_text)  # replace all whitespace with a single whitespace
        cleaned_text = re.sub(r"_+", '_', cleaned_text)  # remove duplicate underscores (from anonymisations)
        cleaned_text = cleaned_text.strip()  # remove leading and trailing whitespace
        cleaned_text = "".join(ch for ch in cleaned_text if unicodedata.category(ch)[0] != "C")  # remove control characters
        return cleaned_text

    def clean_with_functions(self, spider: str, text: str, namespace: dict) -> str:
        """Cleans html documents with cleaning functions"""
        # Parses the html string with bs4 and returns the body content
        soup = bs4.BeautifulSoup(text, "html.parser").find('body')
        if not hasattr(self.cleaning_functions, spider):
            logger.debug(f"There are no special functions for spider {spider}. Just performing default cleaning.")
        else:
            cleaning_function = getattr(self.cleaning_functions, spider)  # retrieve cleaning function by spider
            soup = cleaning_function(soup, namespace)  # invoke cleaning function with soup and namespace

        # we cannot just remove tables because sometimes the content of the entire court decision is inside a table (GL_Omni)
        # for table in soup.find_all("table"):
        #    table.decompose()  # remove all tables from content because they are not useful in raw text
        return soup.get_text()

    def clean_with_regexes(self, spider: str, text: str, namespace: dict) -> str:
        """Cleans pdf documents with cleaning regexes"""
        if spider not in self.spiders_regexes or not self.spiders_regexes[spider]:
            logger.debug(f"There are no special regexes for spider {spider}. Just performing default cleaning.")
            return text
        else:
            regexes = self.spiders_regexes[spider]
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
