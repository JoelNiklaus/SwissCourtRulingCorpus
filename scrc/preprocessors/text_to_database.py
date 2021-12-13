import os
import configparser
import glob
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Optional
from sqlalchemy import MetaData, Table, Column, Integer, String, Date
import pandas as pd
import bs4
import requests
from tqdm.contrib.concurrent import process_map

from root import ROOT_DIR
from scrc.preprocessors.abstract_preprocessor import AbstractPreprocessor
from scrc.utils.language_identification_singleton import LanguageIdentificationSingleton
from scrc.utils.log_utils import get_logger

import tika

from scrc.utils.main_utils import get_config

os.environ['TIKA_LOG_PATH'] = str(AbstractPreprocessor.create_dir(Path(os.getcwd()), 'logs'))
tika.initVM()

from tika import parser

# TODO look at this if db is slow: https://dba.stackexchange.com/questions/151300/improve-update-performance-on-big-table/151316


# TODO if we need to extract data from html with difficult structure such as tables consider using: https://pypi.org/project/inscriptis/

# the keys used in the court dataframes



class TextToDatabase(AbstractPreprocessor):
    """
    Extracts the textual and meta information from the court rulings files and saves it in csv files for each spider
    and in one for all courts combined
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.court_keys = [
            "spider",
            "language",
            "canton",
            "court",
            "chamber",
            "date",
            "file_name",
            "file_number",
            "file_number_additional",
            "html_url",
            "html_raw",
            "pdf_url",
            "pdf_raw",
        ]
        self.lang_id = LanguageIdentificationSingleton()
        self.logger = get_logger(__name__)

    def build_dataset(self) -> None:
        """ Builds the dataset for all the spiders """
        self.logger.info("Started extracting text and metadata from court rulings files")

        processed_file_path = self.progress_dir / "spiders_extracted.txt"
        spider_list, message = self.compute_remaining_spiders(processed_file_path)
        self.logger.info(message)

        for spider in spider_list:
            self.build_spider_dataset(spider)
            self.mark_as_processed(processed_file_path, spider)

        for lang in self.languages:
            self.create_indexes(lang)

        self.logger.info("Finished extracting text and metadata from court rulings files")

    def create_indexes(self, lang):
        self.logger.info(f"Creating indexes for {lang}")
        with self.get_engine(self.db_scrc).connect() as conn:
            for index in self.indexes:
                self.logger.info(f"Creating index for column {index} in table {lang}")
                conn.execute(f"CREATE INDEX IF NOT EXISTS {lang}_{index} ON {lang}({index})")

    def build_spider_dataset(self, spider: str) -> None:
        """ Builds a dataset for a spider """
        spider_dir = self.spiders_dir / spider
        self.logger.info(f"Building spider dataset for {spider}")
        spider_dict_list = self.build_spider_dict_list(spider_dir)

        self.logger.info("Building pandas DataFrame from list of dicts")
        df = pd.DataFrame(spider_dict_list)

        self.logger.info(f"Saving data to db")
        df.to_sql('test', self.get_engine(self.db_scrc), if_exists="append", index=False)

    def build_spider_dict_list(self, spider_dir: Path) -> list:
        """ Builds the spider dict list which we can convert to a pandas Data Frame later """
        # we take the json files as a starting point to get the corresponding html or pdf files
        json_filenames = self.get_filenames_of_extension(spider_dir, 'json')
        spider_dict_list = process_map(self.build_spider_dict, json_filenames, chunksize=1)

        return [spider_dict for spider_dict in spider_dict_list if spider_dict]  # remove None values

    def build_spider_dict(self, json_file: str) -> Optional[dict]:
        """Extracts the information from all the available files"""
        self.logger.debug(f"Processing {json_file}")
        corresponding_pdf_path = Path(json_file).with_suffix('.pdf')
        corresponding_html_path = Path(json_file).with_suffix('.html')
        # if we DO NOT have a court decision corresponding to that found json file
        if not corresponding_html_path.exists() and not corresponding_pdf_path.exists():
            self.logger.warning(f"No court decision found for json file {json_file}")
            return None  # skip the remaining part since we know already that there is no court decision available
        else:
            return self.compose_court_dict(corresponding_html_path, corresponding_pdf_path, json_file)

    def compose_court_dict(self, corresponding_html_path, corresponding_pdf_path, json_file):
        """Composes a court dict from all the available files when we know at least one content file exists"""
        court_dict_template = {key: '' for key in self.court_keys}  # create dict template from court keys

        try:
            general_info = self.extract_general_info(json_file)
        except JSONDecodeError as e:
            self.logger.error(f"Cannot extract metadata from file {json_file}: {e}. Skipping this file")
            return None
        # add general info
        court_dict = dict(court_dict_template, **general_info)

        pdf_content_dict = self.extract_corresponding_pdf_content(corresponding_pdf_path)
        if pdf_content_dict is not None:  # if it could be parsed correctly
            # add pdf content
            court_dict = dict(court_dict, **pdf_content_dict)

        # ATTENTION: If both files exist:
        # html_content_dict will override language from pdf_content_dict because it is more reliable
        html_content_dict = self.extract_corresponding_html_content(corresponding_html_path)
        if html_content_dict is not None:  # if it could be parsed correctly
            # add html content
            court_dict = dict(court_dict, **html_content_dict)  # may override language added from pdf_content_dict
        return court_dict

    def get_filenames_of_extension(self, spider_dir: Path, extension: str) -> list:
        """ Finds all filenames of a given extension in a given directory. """
        filename_list = list(glob.glob(f"{str(spider_dir)}/*.{extension}"))  # Here we can also use regex
        self.logger.info(f"Found {len(filename_list)} {extension} files")
        return filename_list

    def extract_general_info(self, json_file) -> dict:
        """Extracts the filename and spider from the file path and metadata from the json file"""
        self.logger.debug(f"Extracting content from json file: \t {json_file}")
        general_info = {'spider': Path(json_file).parent.name, 'file_name': Path(json_file).stem}
        # loading json content and and extracting relevant metadata
        with open(json_file) as f:
            metadata = json.load(f)
            if 'Signatur' in metadata:
                general_info['canton'] = self.get_canton(metadata['Signatur'])
                general_info['court'] = self.get_court(metadata['Signatur'])
                # the chamber contains all information
                general_info['chamber'] = metadata['Signatur']
            else:
                self.logger.warning("Cannot extract signature from metadata.")
            if 'Num' in metadata:
                file_numbers = metadata['Num']
                if file_numbers:  # if there is at least one entry
                    general_info['file_number'] = file_numbers[0]
                if len(file_numbers) >= 2:  # if there are even two entries (disregard if there are more)
                    # This is to be expected in BVGEer, BGE and BSTG
                    general_info['file_number_additional'] = file_numbers[1]
            else:
                self.logger.warning("Cannot extract file_number from metadata.")
            if 'HTML' in metadata:
                general_info['html_url'] = metadata['HTML']['URL']
            if 'PDF' in metadata:
                general_info['pdf_url'] = metadata['PDF']['URL']
            if 'PDF' not in metadata and 'HTML' not in metadata:
                self.logger.warning("Cannot extract url from metadata.")
            if 'Datum' in metadata:
                general_info['date'] = pd.to_datetime(metadata['Datum'], errors='coerce')
            else:
                self.logger.warning("Cannot extract date from metadata.")
        return general_info

    def get_canton(self, chamber_string):
        # the first two letters always represent the cantonal level (CH for the federation)
        return chamber_string[:2]

    def get_court(self, chamber_string):
        # everything except the last 4 characters represent the court
        return chamber_string[:-4]

    def extract_corresponding_html_content(self, corresponding_html_path) -> Optional[dict]:
        """Extracts the html content, the raw text and the language from the html file, if it exists"""
        if not corresponding_html_path.exists():  # if this court decision is NOT available in html format
            return None
        else:
            self.logger.debug(f"Extracting content from html file: \t {corresponding_html_path}")
            html_raw = corresponding_html_path.read_text()  # get html string
            html_raw = self.remove_nul(html_raw)
            if not html_raw:
                self.logger.error(f"HTML file {corresponding_html_path} is empty.")
                return None
            else:
                soup = bs4.BeautifulSoup(html_raw, "html.parser")  # parse html
                assert soup.find()  # make sure it is valid html
                language = self.lang_id.get_lang(soup.get_text())
                return {"html_raw": html_raw, "language": language}

    def extract_corresponding_pdf_content(self, corresponding_pdf_path) -> Optional[dict]:
        """Extracts the the raw text, the pdf metadata and the language from the pdf file, if it exists"""
        if not corresponding_pdf_path.exists():  # if this court decision is NOT available in pdf format
            return None
        else:
            self.logger.debug(f"Extracting content from pdf file: \t {corresponding_pdf_path}")
            try:
                pdf = parser.from_file(str(corresponding_pdf_path), requestOptions={'timeout': 300})  # parse pdf
            except requests.exceptions.ReadTimeout as e:
                self.logger.error(f"Timeout error occurred for PDF file {corresponding_pdf_path}: {e}")
                return None
            pdf_raw = pdf['content']  # get content
            if not pdf_raw:
                self.logger.error(f"PDF file {corresponding_pdf_path} is empty.")
                return None
            else:
                pdf_raw = self.remove_nul(pdf_raw)
                pdf_raw = pdf_raw.strip()  # strip leading and trailing whitespace
                language = self.lang_id.get_lang(pdf_raw)
                return {"pdf_raw": pdf_raw, "language": language}

    def remove_nul(self, string):
        """Otherwise we get an error when inserting into Postgres"""
        return string.replace("\x00", '')  # remove \x00 (NUL) completely


if __name__ == '__main__':
    config = get_config()

    extractor = TextToDatabase(config)
    extractor.build_dataset()
