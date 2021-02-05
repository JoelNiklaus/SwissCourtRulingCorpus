import configparser
import glob
import json
import multiprocessing
from json import JSONDecodeError
from pathlib import Path
import pandas as pd
import numpy as np
import bs4
from tika import parser

from root import ROOT_DIR
from scrc.language_identification import LanguageIdentification
from scrc.utils.log_utils import get_logger

from multiprocessing import set_start_method

set_start_method("spawn")  # see https://pythonspeed.com/articles/python-multiprocessing/ for more information

logger = get_logger(__name__)

LANGUAGE = LanguageIdentification()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class DataSetBuilder:

    def __init__(self, config: dict):
        self.data_dir = ROOT_DIR / config['dir']['data_dir']
        self.courts_dir = self.data_dir / config['dir']['courts_subdir']  # we get the input from here
        self.csv_dir = self.data_dir / config['dir']['csv_subdir']  # we save the output to here
        self.csv_dir.mkdir(parents=True, exist_ok=True)  # create output folder if it does not exist yet

    def build_dataset(self) -> None:
        """ Builds the dataset for all the courts """
        courts = glob.glob(f"{str(self.courts_dir)}/*")  # Here we can also use regex
        court_list = [Path(court).name for court in courts]
        logger.info(f"Found {len(court_list)} courts")

        # process each court in its own process to speed up this creation by a lot!
        pool = multiprocessing.Pool()
        pool.map(self.build_court_dataset, court_list)
        pool.close()

        # build total df from individual court dfs
        total_df = pd.DataFrame()
        logger.info("Combining all individual court dataframes")
        for court in court_list:
            logger.info(f"Processing court {court}")
            df = pd.read_csv(self.csv_dir / (court + '.csv'))
            total_df = total_df.append(df)

        all_courts_csv_path = self.csv_dir / 'all.csv'
        logger.info(f"Saving all court data to {all_courts_csv_path}")
        total_df.to_csv(all_courts_csv_path)  # also save total df to csv file
        logger.info("Building dataset finished.")

    def build_court_dataset(self, court: str) -> None:
        """ Builds a dataset for a court """
        court_csv_path = self.csv_dir / (court + '.csv')
        if court_csv_path.exists():
            logger.info(f"Skipping court {court}. CSV file already exists.")
            return

        court_dir = self.courts_dir / court
        logger.info(f"Processing {court}")
        court_dict = self.build_court_dict(court_dir)

        logger.info("Building pandas DataFrame from dict")
        df = pd.DataFrame(court_dict)

        logger.info(f"Saving court data to {court_csv_path}")
        df.to_csv(court_csv_path)  # save court to csv

    def build_court_dict(self, court_dir: Path) -> dict:
        """ Builds the court dict which we can convert to a pandas Data Frame later """
        court_dict = {
            "filename": [],
            "court": [],
            "metadata": [],
            "language": [],
            "html_content": [],
            "html_raw": [],
            #        "html_clean": [],
            "pdf_raw": [],
            #        "pdf_clean": [],
            "pdf_metadata": [],
        }

        # we take the json files as a starting point to get the corresponding html or pdf files
        json_filenames = self.get_filenames_of_extension(court_dir, 'json')
        for json_file in json_filenames:
            corresponding_pdf_path = Path(json_file).with_suffix('.pdf')
            corresponding_html_path = Path(json_file).with_suffix('.html')

            # if we have a court decision corresponding to that found json file
            if corresponding_html_path.exists() or corresponding_pdf_path.exists():
                self.handle_general_information(court_dict, json_file)
            else:
                continue  # skip the remaining part since we know already that there is no court decision available

            self.handle_corresponding_pdf_file(corresponding_pdf_path, court_dict)

            # html processing takes place after the pdf processing because it is a bit more reliably
            # and in case both html and pdfs are present (e.g. AG_Gerichte) the language is overwritten here
            self.handle_corresponding_html_file(corresponding_html_path, court_dict)

        return court_dict

    @staticmethod
    def get_filenames_of_extension(court_dir: Path, extension: str) -> list:
        """ Finds all filenames of a given extension in a given directory. """
        filename_list = list(glob.glob(f"{str(court_dir)}/*.{extension}"))  # Here we can also use regex
        logger.info(f"Found {len(filename_list)} {extension} files")
        return filename_list

    @staticmethod
    def handle_general_information(court_dict, json_file):
        """Extracts the filename and the metadata from the json file"""
        logger.debug(f"Extracting content from json file: \t {json_file}")
        court_dict['filename'].append(Path(json_file).stem)
        court_dict['court'].append(Path(json_file).parent.name)
        # loading json content and saving it to 'metadata' key in dict
        with open(json_file) as f:
            try:
                data = json.load(f)
            except JSONDecodeError as e:
                logger.error(f"Error in file {json_file}: ", e)
                logger.info(f"Saving NaN to the metadata column.")
                data = np.nan
            court_dict['metadata'].append(data)

    @staticmethod
    def handle_corresponding_html_file(corresponding_html_path, court_dict):
        """Extracts the html content, the raw text and the language from the html file, if it exists"""
        if corresponding_html_path.exists():  # if this court decision is available in html format
            logger.debug(f"Extracting content from html file: \t {corresponding_html_path}")
            html_str = corresponding_html_path.read_text()  # get html string
            assert html_str is not None and html_str is not ''
            soup = bs4.BeautifulSoup(html_str, "html.parser")  # parse html
            html_raw = soup.get_text()  # extract raw text
            court_dict['html_content'].append(html_str)
            court_dict['html_raw'].append(html_raw)
            court_dict['language'].append(LANGUAGE.get_lang(html_raw))
        else:  # append np.nan values to make sure that lists have equal size
            court_dict['html_content'].append(np.nan)
            court_dict['html_raw'].append(np.nan)

    @staticmethod
    def handle_corresponding_pdf_file(corresponding_pdf_path, court_dict):
        """Extracts the the raw text, the pdf metadata and the language from the pdf file, if it exists"""
        if corresponding_pdf_path.exists():  # if this court decision is available in pdf format
            logger.debug(f"Extracting content from pdf file: \t {corresponding_pdf_path}")
            pdf_bytes = corresponding_pdf_path.read_bytes()
            pdf = parser.from_buffer(pdf_bytes)  # parse pdf
            pdf_raw = pdf['content']  # get content
            if pdf['content'] is not None:
                pdf_raw = pdf_raw.strip()  # strip leading and trailing whitespace
                metadata = pdf['metadata']  # get metadata
                lang = LANGUAGE.get_lang(pdf_raw)
            else:
                logger.error(f"PDF file {corresponding_pdf_path} is empty.")
                pdf_raw, metadata, lang = np.nan, np.nan, np.nan  # set all values to NaN
            court_dict['pdf_raw'].append(pdf_raw)
            court_dict['pdf_metadata'].append(metadata)
            court_dict['language'].append(lang)
        else:  # append np.nan values to make sure that lists have equal size
            court_dict['pdf_raw'].append(np.nan)
            court_dict['pdf_metadata'].append(np.nan)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    data_set_builder = DataSetBuilder(config)
    data_set_builder.build_dataset()
