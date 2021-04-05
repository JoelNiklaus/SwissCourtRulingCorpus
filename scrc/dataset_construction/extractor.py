import configparser
import glob
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Optional

import pandas as pd

import bs4
import requests
from tika import parser
from tqdm.contrib.concurrent import process_map

from root import ROOT_DIR
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.language_identification import LanguageIdentification
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import court_keys

LANG_ID = LanguageIdentification()


class Extractor(DatasetConstructorComponent):
    """
    Extracts the textual and meta information from the court rulings files and saves it in csv files for each spider
    and in one for all courts combined
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

    def build_dataset(self) -> None:
        """ Builds the dataset for all the spiders """
        spider_list = [Path(spider).name for spider in glob.glob(f"{str(self.spiders_dir)}/*")]

        for spider in spider_list:
            self.build_spider_dataset(spider)

        self.logger.info("Building dataset finished.")

    def build_spider_dataset(self, spider: str) -> None:
        """ Builds a dataset for a spider """
        spider_csv_path = self.raw_csv_subdir / (spider + '.csv')
        if spider_csv_path.exists():
            self.logger.info(f"Skipping spider {spider}. CSV file already exists.")
            return

        spider_dir = self.spiders_dir / spider
        self.logger.info(f"Building spider dataset for {spider}")
        spider_dict_list = self.build_spider_dict_list(spider_dir)

        self.logger.info("Building pandas DataFrame from list of dicts")
        df = pd.DataFrame(spider_dict_list)

        self.logger.info(f"Saving data to {spider_csv_path}")
        df.to_csv(spider_csv_path, index=False)  # save spider to csv

    def build_spider_dict_list(self, spider_dir: Path) -> list:
        """ Builds the spider dict list which we can convert to a pandas Data Frame later """
        # we take the json files as a starting point to get the corresponding html or pdf files
        json_filenames = self.get_filenames_of_extension(spider_dir, 'json')
        spider_dict_list = process_map(self.build_spider_dict, json_filenames, chunksize=100)

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
            try:
                metadata = json.load(f)
                if 'Signatur' in metadata:
                    # the first two letters always represent the cantonal level (CH for the federation)
                    general_info['canton'] = metadata['Signatur'][:2]
                    # everything except the last 4 characters represent the court
                    general_info['court'] = metadata['Signatur'][:-4]
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
                    general_info['date'] = metadata['Datum']
                else:
                    self.logger.warning("Cannot extract date from metadata.")
            except JSONDecodeError as e:
                self.logger.error(f"Error in file {json_file}: {e}. Cannot extract file_number, url and date.")
        return general_info

    def extract_corresponding_html_content(self, corresponding_html_path) -> Optional[dict]:
        """Extracts the html content, the raw text and the language from the html file, if it exists"""
        if not corresponding_html_path.exists():  # if this court decision is NOT available in html format
            return None
        else:
            self.logger.debug(f"Extracting content from html file: \t {corresponding_html_path}")
            html_raw = corresponding_html_path.read_text()  # get html string
            assert html_raw is not None and html_raw != ''
            soup = bs4.BeautifulSoup(html_raw, "html.parser")  # parse html
            assert soup.find()  # make sure it is valid html
            language = LANG_ID.get_lang(soup.get_text())
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
            if pdf['content'] is None:
                self.logger.error(f"PDF file {corresponding_pdf_path} is empty.")
                return None
            else:
                pdf_raw = pdf_raw.strip()  # strip leading and trailing whitespace
                language = LANG_ID.get_lang(pdf_raw)
                return {"pdf_raw": pdf_raw, "language": language}


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    extractor = Extractor(config)
    extractor.build_dataset()
