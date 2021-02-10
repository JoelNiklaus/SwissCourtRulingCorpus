import configparser
import glob
import json
import multiprocessing
from json import JSONDecodeError
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import bs4
from tika import parser

from root import ROOT_DIR
from scrc.language_identification import LanguageIdentification
from scrc.utils.log_utils import get_logger

logger = get_logger(__name__)

LANGUAGE = LanguageIdentification()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class DataSetBuilder:
    """
    Extracts the textual and meta information from the court rulings files and saves it in csv files for each court
    and in one for all courts combined
    """

    def __init__(self, config: dict):
        self.data_dir = ROOT_DIR / config['dir']['data_dir']
        self.courts_dir = self.data_dir / config['dir']['courts_subdir']  # we get the input from here
        self.csv_dir = self.data_dir / config['dir']['csv_subdir']  # we save the output to here
        self.csv_dir.mkdir(parents=True, exist_ok=True)  # create output folder if it does not exist yet
        self.all_courts_csv_path = self.csv_dir / '_all.csv'

    def build_dataset(self) -> None:
        """ Builds the dataset for all the courts """
        courts = glob.glob(f"{str(self.courts_dir)}/*")  # Here we can also use regex
        court_list = [Path(court).name for court in courts]
        logger.info(f"Found {len(court_list)} courts")

        # process each court in its own process to speed up this creation by a lot!
        pool = multiprocessing.Pool()
        pool.map(self.build_court_dataset, court_list)
        pool.close()

        self.combine_courts(court_list)
        logger.info("Building dataset finished.")

    def combine_courts(self, court_list) -> None:
        """build total df from individual court dfs"""
        if self.all_courts_csv_path.exists():  # Delete the combined file in order to recreate it.
            logger.info(f"Combined csv file already exists at {self.all_courts_csv_path}")
            return
        logger.info(f"Combining all individual court dataframes and saving to {self.all_courts_csv_path}")
        for i, court in enumerate(court_list):
            mode, header = 'a', False  # normally append and don't add column names
            if i == 0:  # if we are processing the first court
                mode, header = 'w', True  # for the first one: normal write mode and add column names
            logger.info(f"Processing court {court}")
            df = pd.read_csv(self.csv_dir / (court + '.csv'))
            df.to_csv(self.all_courts_csv_path, mode=mode, header=header, index=False)

    def build_court_dataset(self, court: str) -> None:
        """ Builds a dataset for a court """
        court_csv_path = self.csv_dir / (court + '.csv')
        if court_csv_path.exists():
            logger.info(f"Skipping court {court}. CSV file already exists.")
            return

        court_dir = self.courts_dir / court
        logger.info(f"Processing {court}")
        court_dict_list = self.build_court_dict_list(court_dir)

        logger.info("Building pandas DataFrame from list of dicts")
        df = pd.DataFrame(court_dict_list)

        logger.info(f"Saving court data to {court_csv_path}")
        df.to_csv(court_csv_path, index=False)  # save court to csv

    def build_court_dict_list(self, court_dir: Path) -> list:
        """ Builds the court dict list which we can convert to a pandas Data Frame later """
        court_dict_list = []

        # we take the json files as a starting point to get the corresponding html or pdf files
        json_filenames = self.get_filenames_of_extension(court_dir, 'json')
        for json_file in json_filenames:
            court_dict = self.build_court_dict(json_file)
            if court_dict:
                court_dict_list.append(court_dict)

        return court_dict_list

    def build_court_dict(self, json_file: str) -> Optional[dict]:
        """Extracts the information from all the available files"""
        corresponding_pdf_path = Path(json_file).with_suffix('.pdf')
        corresponding_html_path = Path(json_file).with_suffix('.html')
        # if we DO NOT have a court decision corresponding to that found json file
        if not corresponding_html_path.exists() and not corresponding_pdf_path.exists():
            logger.warning(f"No court decision found for json file {json_file}")
            return None  # skip the remaining part since we know already that there is no court decision available
        else:
            court_dict_template = {
                "filename": '',
                "court": '',
                "metadata": '',
                "language": '',
                "html_content": '',
                "html_raw": '',
                "html_clean": '',  # will remain empty for now
                "pdf_raw": '',
                "pdf_clean": '',  # will remain empty for now
            }
            general_info = self.extract_general_info(json_file)

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

    @staticmethod
    def get_filenames_of_extension(court_dir: Path, extension: str) -> list:
        """ Finds all filenames of a given extension in a given directory. """
        filename_list = list(glob.glob(f"{str(court_dir)}/*.{extension}"))  # Here we can also use regex
        logger.info(f"Found {len(filename_list)} {extension} files")
        return filename_list

    @staticmethod
    def extract_general_info(json_file) -> dict:
        """Extracts the filename and the metadata from the json file"""
        logger.debug(f"Extracting content from json file: \t {json_file}")
        filename = Path(json_file).stem
        court = Path(json_file).parent.name
        # loading json content and saving it to 'metadata' key in dict
        with open(json_file) as f:
            try:
                metadata = json.load(f)
            except JSONDecodeError as e:
                logger.error(f"Error in file {json_file}: ", e)
                logger.info(f"Saving NaN to the metadata column.")
                metadata = ''
        return {"filename": filename, "court": court, "metadata": metadata}

    @staticmethod
    def extract_corresponding_html_content(corresponding_html_path) -> Optional[dict]:
        """Extracts the html content, the raw text and the language from the html file, if it exists"""
        if not corresponding_html_path.exists():  # if this court decision is NOT available in html format
            return None
        else:
            logger.debug(f"Extracting content from html file: \t {corresponding_html_path}")
            html_content = corresponding_html_path.read_text()  # get html string
            assert html_content is not None and html_content != ''
            soup = bs4.BeautifulSoup(html_content, "html.parser")  # parse html
            html_raw = soup.get_text()  # extract raw text
            language = LANGUAGE.get_lang(html_raw)
            return {"html_content": html_content, "html_raw": html_raw, "language": language}

    @staticmethod
    def extract_corresponding_pdf_content(corresponding_pdf_path) -> Optional[dict]:
        """Extracts the the raw text, the pdf metadata and the language from the pdf file, if it exists"""
        if not corresponding_pdf_path.exists():  # if this court decision is NOT available in pdf format
            return None
        else:
            logger.debug(f"Extracting content from pdf file: \t {corresponding_pdf_path}")
            pdf_bytes = corresponding_pdf_path.read_bytes()
            pdf = parser.from_buffer(pdf_bytes)  # parse pdf
            pdf_raw = pdf['content']  # get content
            if pdf['content'] is None:
                logger.error(f"PDF file {corresponding_pdf_path} is empty.")
                return None
            else:
                pdf_raw = pdf_raw.strip()  # strip leading and trailing whitespace
                language = LANGUAGE.get_lang(pdf_raw)
                return {'pdf_raw': pdf_raw, 'language': language}


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    data_set_builder = DataSetBuilder(config)
    data_set_builder.build_dataset()
