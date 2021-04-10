import configparser
from pathlib import Path

import bs4
import glob
import requests
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from root import ROOT_DIR
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import save_to_path

base_url = "https://entscheidsuche.ch/"

supported_suffixes = ['.htm', '.html', '.pdf', '.txt', '.json']
supported_languages = ['de', 'fr', 'it']
excluded_link_names = ['Name', 'Last modified', 'Size', 'Description', 'Parent Directory', 'Index', 'Jobs', 'Sitemaps']

# TODO check everywhere per file if there are new decisions, not per spider, so that we can easily download the most recent data

class Scraper(DatasetConstructorComponent):
    """Scrapes the court rulings with the associated metadata files from entscheidsuche.ch/docs"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

    def download_subfolders(self, url: str):
        """
        Download entire subfolders recursively
        :param url:
        :return:
        """
        self.logger.info(f"Started downloading from {url}")
        r = requests.get(url)  # get starting page
        data = bs4.BeautifulSoup(r.text, "html.parser")  # parse html
        links = data.find_all("a")  # find all links

        included_links = [Path(link["href"]) for link in links if not self.link_is_excluded(link.text)]
        self.logger.info(f"Found {len(included_links)} links in total")

        already_downloaded_links = [Path(spider).stem for spider in glob.glob(f"{str(self.spiders_dir)}/*")]
        self.logger.info(f"Found {len(already_downloaded_links)} links already downloaded: {already_downloaded_links}")

        link_already_downloaded = lambda link: any(downloaded in str(link) for downloaded in already_downloaded_links)
        links_still_to_download = [link for link in included_links if not link_already_downloaded(link)]
        self.logger.info(f"Found {len(links_still_to_download)} links still to download")

        for link in links_still_to_download:
            self.download_files(link)

        self.logger.info(f"Finished downloading from {url}")

    def link_is_excluded(self, link_text: str):
        """ Exclude links other than the folders to the courts """
        if '.' in link_text:  # exclude filenames
            return True
        if len(link_text) < 3:  # exclude . and ..
            return True
        for excluded in excluded_link_names:
            if excluded in link_text:  # exclude blacklisted link names
                return True
        return False

    def download_files(self, sub_folder: Path):
        """
        Download files from entscheidsuche

        :param sub_folder:
        :return:
        """
        self.logger.info(f"Started downloading from {sub_folder} ...")
        r = requests.get(f"{base_url}/{sub_folder}")  # get starting page
        data = bs4.BeautifulSoup(r.text, "html.parser")  # parse html
        links = data.find_all("a")  # find all links
        included_links = [Path(link["href"]) for link in links if Path(link["href"]).suffix in supported_suffixes]
        self.logger.info(f"Found {len(included_links)} links")

        # process each court in its own process to speed up this creation by a lot!
        # in case that the server has not enough capacity (we are encounterring 'Connection reset by peer')
        # => decrease number of processes
        process_map(self.download_file_from_url, included_links, max_workers=8, chunksize=100)

        # for link in tqdm(included_links):
        #    self.download_file_from_url(link)

        self.logger.info(f"Finished downloading from {sub_folder} ...")

    def download_file_from_url(self, url):
        """download the file from a link and save it"""
        # can lead to problems inside multiprocessing
        # => would need concurrent log handler: https://pypi.org/project/ConcurrentLogHandler/
        # self.logger.debug(f"Downloading from {url}")
        try:
            r = requests.get(base_url + str(url))  # make request to download file
            # save to the last two parts of the url (folder and filename)
            save_to_path(r.content, self.spiders_dir / Path(*url.parts[-2:]))
        except Exception as e:
            self.logger.error(f"Caught an exception while processing {str(url)}\n{e}")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    scraper = Scraper(config)
    scraper.download_subfolders(base_url + "docs/")
