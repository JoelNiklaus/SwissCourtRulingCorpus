import json
from pathlib import Path
import re
import unicodedata
import configparser
from os.path import exists
from root import ROOT_DIR
import pandas as pd

from scrc.utils.log_utils import get_logger

logger = get_logger()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def save_to_path(content, path, overwrite=False):
    """
    Create the parent directories of they do not exist.
    If file does not exist already, save content to path.
    :param content:     content to be saved
    :param path:        path of file to be saved
    :param overwrite:   if True overwrites the current content at that path
    :return:
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # check if
    if path.exists():
        logger.debug(f"Path {path} exists already")
        if overwrite:
            logger.debug("Overwrite option specified")
        else:
            return

    logger.debug(f"Saving file to {path}")
    # actually do the saving
    if isinstance(content, bytes):
        path.write_bytes(content)
    elif isinstance(content, str):
        path.write_text(content)
    elif isinstance(content, dict):
        path.write_text(json.dumps(content, indent = 4))
    else:
        raise ValueError(f"Invalid data type {type(content)} supplied.")

    
def get_paragraphs_unified(decision):
    if isinstance(decision, str):
        return get_pdf_paragraphs(decision)
    elif decision:  
        paragraphs = []
        for string in decision.strings:
            paragraphs.append(string) 
        paragraphs = list(filter(None, map(clean_text, paragraphs)))
        return paragraphs
    
def clean_whitespace(str):
    str = str.strip()
    if str:
        return True
    return False

def get_pdf_paragraphs(soup: str) -> list:
    """
    Get the paragraphs of a decision
    :param soup:    the string extracted of the pdf
    :return:        a list of paragraphs
    """

    paragraphs = []
    # remove spaces between two line breaks
    soup = re.sub('\\n +\\n', '\\n\\n', soup)
    # split the lines when there are two line breaks
    lines = soup.split('\n\n')
    for element in lines:
        element = element.replace('  ', ' ')
        paragraph = clean_text(element)
        if paragraph not in ['', ' ', None]:  # discard empty paragraphs
            paragraphs.append(paragraph)
    return paragraphs


def get_raw_text(html) -> str:
    """
    Add the entire text: harder for doing sentence splitting later because of header and footer
    :param html:
    :return:
    """

    raw_text = html.get_text()
    return raw_text


def clean_text(text: str) -> str:
    """
    Clean text from nasty tokens
    :param text:    the text to be cleaned
    :return:
    """
    if not text:
        return ''
    cleaned_text = text
    # https://stackoverflow.com/questions/16467479/normalizing-unicode
    cleaned_text = unicodedata.normalize(
        'NFKC', cleaned_text)  # normalize strings
    # remove hyphens before new line
    cleaned_text = re.sub('(\w+)-\n+(\w+)', '\1\2', cleaned_text)
    # replace NBSP with normal whitespace
    cleaned_text = re.sub(r"\u00a0", ' ', cleaned_text)
    # replace \xa0 with normal whitespace
    cleaned_text = re.sub(r"\xa0", ' ', cleaned_text)
    cleaned_text = re.sub(r"\x00", '', cleaned_text)  # remove \x00 completely
    # replace all whitespace with a single whitespace
    cleaned_text = re.sub(r"\s+", ' ', cleaned_text)
    # remove duplicate underscores (from anonymisations)
    cleaned_text = re.sub(r"_+", '_', cleaned_text)
    cleaned_text = cleaned_text.strip()  # remove leading and trailing whitespace
    cleaned_text = "".join(
        ch for ch in cleaned_text if unicodedata.category(ch)[0] != "C")  # remove control characters
    return cleaned_text


def chunker(iterable, chunk_size):
    return (iterable[pos: pos + chunk_size] for pos in range(0, len(iterable), chunk_size))


def get_file_gen(path):
    def get_path(path, chunk):
        return path / f"part.{chunk}.parquet"

    chunk = 0
    file = get_path(path, chunk)
    while file.exists():
        yield chunk, pd.read_parquet(file)
        chunk += 1
        file = get_path(path, chunk)
    return None


def string_contains_one_of_list(string: str, lst: list):
    """
    If the string contains an item in the list,
     we return that item (which can also be used as a truth value in if conditions)
     and otherwise we return False
     """
    for item in lst:
        if item in string:
            return item
    return False


def int_to_roman(num: int) -> str:
    """https://www.w3resource.com/python-exercises/class-exercises/python-class-exercise-1.php
    Converts an integer to a roman numeral string
    :param num: the input number
    :return:    the output roman numeral string"""
    lookup = [
        (1000, 'M'),
        (900, 'CM'),
        (500, 'D'),
        (400, 'CD'),
        (100, 'C'),
        (90, 'XC'),
        (50, 'L'),
        (40, 'XL'),
        (10, 'X'),
        (9, 'IX'),
        (5, 'V'),
        (4, 'IV'),
        (1, 'I'),
    ]
    res = ''
    for (n, roman) in lookup:
        (d, num) = divmod(num, n)
        res += roman * d
    return res


def roman_to_int(s: str) -> int:
    """https://www.w3resource.com/python-exercises/class-exercises/python-class-exercise-2.php
    Converts a roman numeral string to an integer
    :param num: the input roman numeral string
    :return:    the output number"""
    rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    int_val = 0
    for i in range(len(s)):
        if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
            int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
        else:
            int_val += rom_val[s[i]]
    return int_val


def get_config() -> configparser.ConfigParser:
    """Returns the parsed `config.ini` / `rootconfig.ini` files"""
    config = configparser.ConfigParser()
    # this stops working when the script is called from the src directory!
    config.read(ROOT_DIR / 'config.ini')
    if exists(ROOT_DIR / 'rootconfig.ini'):
        # this stops working when the script is called from the src directory!
        config.read(ROOT_DIR / 'rootconfig.ini')
    return config


def save_df_to_cache(df: pd.DataFrame, path: Path):
    logger.info(f"Writing data to cache at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="gzip")


def retrieve_from_cache_if_exists(path: Path):
    if path.exists():
        logger.info(f"Loading data from cache at {path}")
        return pd.read_parquet(path)   # index_col=False gives error

    else:
        return pd.DataFrame([])
