import configparser
from pathlib import Path

import bs4
import requests

from root import ROOT_DIR
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import save_to_path

base_url = "https://entscheidsuche.ch/"

supported_suffixes = ['.htm', '.html', '.pdf', '.txt', '.json']
supported_languages = ['de', 'fr', 'it']
excluded_link_names = ['Name', 'Last modified', 'Size', 'Description', 'Parent Directory', 'Index', 'Jobs']

logger = get_logger(__name__)


def download_subfolders(url: str, data_dir: Path):
    """
    Download entire subfolders recursively
    :param url:
    :return:
    """
    logger.info(f"Downloading from {url}")
    r = requests.get(url)  # get starting page
    data = bs4.BeautifulSoup(r.text, "html.parser")  # parse html
    links = data.find_all("a")  # find all links
    logger.info(f"Found {len(links)} links")
    for link in links:
        if not link_is_excluded(link.text):
            download_files(Path(link["href"]), data_dir)  # download all files in that subfolder


def link_is_excluded(link_text: str):
    """ Exclude links other than the folders to the courts """
    if '.' in link_text:  # exclude filenames
        return True
    if len(link_text) < 3:  # exclude . and ..
        return True
    for excluded in excluded_link_names:
        if excluded in link_text:  # exclude blacklisted link names
            return True
    return False


def download_files(sub_folder: Path, data_dir: Path):
    """
    Download files from entscheidsuche

    :param sub_folder:
    :return:
    """
    logger.info(f"Downloading from {sub_folder} ...")
    r = requests.get(f"{base_url}/{sub_folder}")  # get starting page
    data = bs4.BeautifulSoup(r.text, "html.parser")  # parse html
    links = data.find_all("a")  # find all links
    logger.info(f"Found {len(links)} links")
    for link in links:
        url = Path(link["href"])  # get link url
        if url.suffix in supported_suffixes:  # only links to files
            try:
                r = requests.get(base_url + str(url))  # make request to download file
                # save to the last two parts of the url (folder and filename)
                save_to_path(r.content, data_dir / Path(*url.parts[-2:]))
            except Exception as e:
                logger.error(f"Caught an exception while processing {str(url)}\n{e}")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!
    data_dir = ROOT_DIR / config['dir']['data_dir']
    courts_dir = data_dir / config['dir']['courts_subdir']

    download_subfolders(base_url + "docs/", courts_dir)
