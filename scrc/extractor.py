import configparser
import glob
import json
import multiprocessing
from json import JSONDecodeError
from pathlib import Path
from typing import Optional

import pandas as pd

import bs4
import requests
from tika import parser

from root import ROOT_DIR
from scrc.language_identification import LanguageIdentification
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import court_keys

logger = get_logger(__name__)

LANGUAGE = LanguageIdentification()


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



class Extractor:
    """
    Extracts the textual and meta information from the court rulings files and saves it in csv files for each court
    and in one for all courts combined
    """

    def __init__(self, config: dict):
        self.data_dir = ROOT_DIR / config['dir']['data_dir']
        self.courts_dir = self.data_dir / config['dir']['courts_subdir']  # we get the input from here
        self.csv_dir = self.data_dir / config['dir']['csv_subdir']
        self.raw_csv_subdir = self.csv_dir / config['dir']['raw_csv_subdir']  # we save the output to here
        self.raw_csv_subdir.mkdir(parents=True, exist_ok=True)  # create output folder if it does not exist yet

    def build_dataset(self) -> None:
        """ Builds the dataset for all the courts """
        court_list = [Path(court).name for court in glob.glob(f"{str(self.courts_dir)}/*")]
        for court in court_list:
            self.build_court_dataset(court)

        logger.info("Building dataset finished.")

    def build_court_dataset(self, court: str) -> None:
        """ Builds a dataset for a court """
        court_csv_path = self.raw_csv_subdir / (court + '.csv')
        if court_csv_path.exists():
            logger.info(f"Skipping court {court}. CSV file already exists.")
            return

        court_dir = self.courts_dir / court
        logger.info(f"Building court dataset for {court}")
        court_dict_list = self.build_court_dict_list(court_dir)

        logger.info("Building pandas DataFrame from list of dicts")
        df = pd.DataFrame(court_dict_list)

        logger.info(f"Saving court data to {court_csv_path}")
        df.to_csv(court_csv_path, index=False)  # save court to csv

    def build_court_dict_list(self, court_dir: Path) -> list:
        """ Builds the court dict list which we can convert to a pandas Data Frame later """
        # we take the json files as a starting point to get the corresponding html or pdf files
        json_filenames = self.get_filenames_of_extension(court_dir, 'json')
        with multiprocessing.Pool() as pool:
            court_dict_list = pool.map(self.build_court_dict, json_filenames)

        return [court_dict for court_dict in court_dict_list if court_dict]  # remove None values

    def build_court_dict(self, json_file: str) -> Optional[dict]:
        """Extracts the information from all the available files"""
        logger.info(f"Processing {json_file}")
        corresponding_pdf_path = Path(json_file).with_suffix('.pdf')
        corresponding_html_path = Path(json_file).with_suffix('.html')
        # if we DO NOT have a court decision corresponding to that found json file
        if not corresponding_html_path.exists() and not corresponding_pdf_path.exists():
            logger.warning(f"No court decision found for json file {json_file}")
            return None  # skip the remaining part since we know already that there is no court decision available
        else:
            return self.compose_court_dict(corresponding_html_path, corresponding_pdf_path, json_file)

    def compose_court_dict(self, corresponding_html_path, corresponding_pdf_path, json_file):
        """Composes a court dict from all the available files when we know at least one content file exists"""
        court_dict_template = {key: '' for key in court_keys}  # create dict template from court keys

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
        """Extracts the filename and court from the file path and metadata from the json file"""
        logger.debug(f"Extracting content from json file: \t {json_file}")
        general_info = {'spider': Path(json_file).parent.name, 'file_name': Path(json_file).stem}
        # loading json content and and extracting relevant metadata
        with open(json_file) as f:
            try:
                metadata = json.load(f)
                if 'Signatur' in metadata:
                    # the first two letters always represent the cantonal level (CH for the federation)
                    general_info['canton'] = metadata['Signatur'][:2]
                    # everything except the last 4 characters represent the court
                    general_info['court'] = metadata['Signatur'][:-4]
                    general_info['chamber'] = metadata['Signatur']
                else:
                    logger.warning("Cannot extract signature from metadata.")
                if 'Num' in metadata:
                    file_numbers = metadata['Num']
                    if file_numbers:  # if there is at least one entry
                        general_info['file_number'] = file_numbers[0]
                    if len(file_numbers) >= 2:  # if there are even two entries (disregard if there are more)
                        # This is to be expected in BVGEer, BGE and BSTG
                        general_info['file_number_additional'] = file_numbers[1]
                else:
                    logger.warning("Cannot extract file_number from metadata.")
                if 'PDF' in metadata:
                    general_info['url'] = metadata['PDF']['URL']
                elif 'HTML' in metadata:
                    general_info['url'] = metadata['HTML']['URL']
                else:
                    logger.warning("Cannot extract url from metadata.")
                if 'Datum' in metadata:
                    general_info['date'] = metadata['Datum']
                else:
                    logger.warning("Cannot extract date from metadata.")
            except JSONDecodeError as e:
                logger.error(f"Error in file {json_file}: {e}. Cannot extract file_number, url and date.")
        return general_info

    @staticmethod
    def extract_corresponding_html_content(corresponding_html_path) -> Optional[dict]:
        """Extracts the html content, the raw text and the language from the html file, if it exists"""
        if not corresponding_html_path.exists():  # if this court decision is NOT available in html format
            return None
        else:
            logger.debug(f"Extracting content from html file: \t {corresponding_html_path}")
            html_raw = corresponding_html_path.read_text()  # get html string
            assert html_raw is not None and html_raw != ''
            soup = bs4.BeautifulSoup(html_raw, "html.parser")  # parse html
            assert soup.find()  # make sure it is valid html
            language = LANGUAGE.get_lang(soup.get_text())
            return {"html_raw": html_raw, "language": language}

    @staticmethod
    def extract_corresponding_pdf_content(corresponding_pdf_path) -> Optional[dict]:
        """Extracts the the raw text, the pdf metadata and the language from the pdf file, if it exists"""
        if not corresponding_pdf_path.exists():  # if this court decision is NOT available in pdf format
            return None
        else:
            logger.debug(f"Extracting content from pdf file: \t {corresponding_pdf_path}")
            try:
                pdf = parser.from_file(str(corresponding_pdf_path), requestOptions={'timeout': 300})  # parse pdf
            except requests.exceptions.ReadTimeout as e:
                logger.error(f"Timeout error occurred for PDF file {corresponding_pdf_path}: {e}")
                return None
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

    extractor = Extractor(config)
    extractor.build_dataset()
