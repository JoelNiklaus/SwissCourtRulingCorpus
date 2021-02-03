import configparser
import glob
import json
from pathlib import Path
import pandas as pd
import numpy as np
import bs4
from tika import parser

from root import ROOT_DIR
from scrc.lang_id import LanguageIdentification
from scrc.utils.log_utils import get_logger

logger = get_logger(__name__)

LANGUAGE = LanguageIdentification()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def build_dataset(data_dir: Path) -> pd.DataFrame:
    """ Builds the dataset for all the courts """
    court_list = list(glob.glob(f"{str(data_dir)}/*"))  # Here we can also use regex
    logger.info(f"Found {len(court_list)} courts")
    total_df = pd.DataFrame()
    for court in court_list:
        df = build_court_dataset(data_dir, Path(court).name)
        court_csv_path = data_dir / (court + '.csv')
        logger.info(f"Saved court data to {court_csv_path}")
        df.to_csv(court_csv_path)  # save court to csv
        total_df = total_df.append(df)
    return total_df


def build_court_dataset(data_dir: Path, court: str) -> pd.DataFrame:
    """ Builds a dataset for a court """
    court_dir = data_dir / court
    logger.info(f"Processing {court}")
    court_dict = build_court_dict(court_dir)

    logger.info("Building pandas DataFrame from dict")
    return pd.DataFrame(court_dict)


def build_court_dict(court_dir: Path) -> dict:
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
    json_filenames = get_filenames_of_extension(court_dir, 'json')
    for json_file in json_filenames:
        corresponding_pdf_path = Path(json_file).with_suffix('.pdf')
        corresponding_html_path = Path(json_file).with_suffix('.html')

        # if we have a court decision corresponding to that found json file
        if corresponding_html_path.exists() or corresponding_pdf_path.exists():
            handle_general_information(court_dict, json_file)
        else:
            continue  # skip the remaining part since we know already that there is no court decision available

        handle_corresponding_pdf_file(corresponding_pdf_path, court_dict)

        # html processing takes place after the pdf processing because it is a bit more reliably
        # and in case both html and pdfs are present (e.g. AG_Gerichte) the language is overwritten here
        handle_corresponding_html_file(corresponding_html_path, court_dict)

    return court_dict


def handle_general_information(court_dict, json_file):
    """Extracts the filename and the metadata from the json file"""
    court_dict['filename'].append(Path(json_file).stem)
    court_dict['court'].append(Path(json_file).parent.name)
    # loading json content and saving it to 'metadata' key in dict
    with open(json_file) as f:
        data = json.load(f)
        court_dict['metadata'].append(data)


def handle_corresponding_html_file(corresponding_html_path, court_dict):
    """Extracts the html content, the raw text and the language from the html file, if it exists"""
    if corresponding_html_path.exists():  # if this court decision is available in html format
        logger.info(f"Extracting content from html file: \t {corresponding_html_path}")
        html_str = corresponding_html_path.read_text()  # get html string
        soup = bs4.BeautifulSoup(html_str, "html.parser")  # parse html
        html_raw = soup.get_text()  # extract raw text
        court_dict['html_content'].append(html_str)
        court_dict['html_raw'].append(html_raw)
        court_dict['language'].append(LANGUAGE.get_lang(html_raw))
    else:  # append np.nan values to make sure that lists have equal size
        court_dict['html_content'].append(np.nan)
        court_dict['html_raw'].append(np.nan)


def handle_corresponding_pdf_file(corresponding_pdf_path, court_dict):
    """Extracts the the raw text, the pdf metadata and the language from the pdf file, if it exists"""
    if corresponding_pdf_path.exists():  # if this court decision is available in pdf format
        logger.info(f"Extracting content from pdf file: \t {corresponding_pdf_path}")
        pdf_bytes = corresponding_pdf_path.read_bytes()
        pdf = parser.from_buffer(pdf_bytes)  # parse pdf
        pdf_raw = pdf['content'].strip()  # get content and strip leading and trailing whitespace
        court_dict['pdf_raw'].append(pdf_raw)
        court_dict['pdf_metadata'].append(pdf['metadata'])  # get metadata
        court_dict['language'].append(LANGUAGE.get_lang(pdf_raw))
    else:  # append np.nan values to make sure that lists have equal size
        court_dict['pdf_raw'].append(np.nan)
        court_dict['pdf_metadata'].append(np.nan)


def get_filenames_of_extension(court_dir: Path, extension: str) -> list:
    """ Finds all filenames of a given extension in a given directory. """
    filename_list = list(glob.glob(f"{str(court_dir)}/*.{extension}"))  # Here we can also use regex
    logger.info(f"Found {len(filename_list)} {extension} files")
    return filename_list


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!
    data_dir = ROOT_DIR / config['dir']['data_dir']

    df = build_dataset(data_dir)
    df.to_csv(data_dir / 'all.csv')
    logger.info("Building dataset finished.")
