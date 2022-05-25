import urllib.parse
from pathlib import Path

# IMPORTANT: absolutely necessary to speed up bs4 parsing: https://thehftguy.com/2020/07/28/making-beautifulsoup-parsing-10-times-faster/
# TODO use everywhere where we need to parse with bs4
import cchardet
import lxml

import bs4
import glob
import requests
from tqdm.contrib.concurrent import process_map

from scrc.preprocessors.abstract_preprocessor import AbstractPreprocessor
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config, save_to_path

base_url = "https://entscheidsuche.ch/"

supported_suffixes = ['.htm', '.html', '.pdf', '.txt', '.json']
supported_languages = ['de', 'fr', 'it']
excluded_link_names = ['Name', 'Last modified', 'Size', 'Description', 'Parent Directory', 'Index', 'Jobs', 'Sitemaps']


class Scraper(AbstractPreprocessor):
    """Scrapes the court rulings with the associated metadata files from entscheidsuche.ch/docs"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

    def download_subfolders(self, url: str):
        """
        Download entire subfolders recursively
        :param url:
        :return: list of paths of new downloaded files
        """
        self.logger.info(f"Started downloading from {url}")
        r = requests.get(url)  # get starting page
        data = bs4.BeautifulSoup(r.text, "lxml")  # parse html
        links = data.find_all("a")  # find all links

        included_links = [Path(link["href"]) for link in links if not self.link_is_excluded(link.text)]
        self.logger.info(f"Found {len(included_links)} spiders/folders in total")

        all_new_files: list[Path] = []
        for link in included_links:
            new_files = self.download_files(link)
            all_new_files.extend(new_files)

        self.logger.info(f"Finished downloading from {url}")

        return all_new_files

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
        data = bs4.BeautifulSoup(r.text, "lxml")  # parse html
        links = data.find_all("a")  # find all links
        included_links = [Path(link["href"]) for link in links if Path(link["href"]).suffix in supported_suffixes]
        self.logger.info(f"Found {len(included_links)} links")

        if self.process_new_files_only:
            already_downloaded_links = [Path(spider).stem for spider in
                                        glob.glob(f"{str(self.spiders_dir)}/{sub_folder.stem}/*")]
            self.logger.info(f"{len(already_downloaded_links)} of {len(included_links)} are already downloaded")

            if len(included_links) == len(already_downloaded_links):
                links_still_to_download = []  # shortcut to save computation
            else:
                already_downloaded_links = set(already_downloaded_links)  # set for faster lookup
                links_still_to_download = [link for link in included_links if not link.stem in already_downloaded_links]
        else:
            links_still_to_download = included_links

        self.logger.info(f"Downloading {len(links_still_to_download)} links")

        # process each court in its own process to speed up this creation by a lot!
        # in case that the server has not enough capacity (we are encountering 'Connection reset by peer')
        # => decrease number of processes
        if links_still_to_download:
            process_map(self.download_file_from_url, links_still_to_download, max_workers=8, chunksize=100)

        # for link in tqdm(included_links):
        #    self.download_file_from_url(link)

        self.logger.info(f"Finished downloading from {sub_folder} ...")
        return links_still_to_download

    def download_file_from_url(self, url):
        """download the file from a link and save it"""
        # can lead to problems inside multiprocessing
        # => would need concurrent log handler: https://pypi.org/project/ConcurrentLogHandler/
        # self.logger.debug(f"Downloading from {url}")
        try:
            r = requests.get(base_url + str(url))  # make request to download file
            # save to the last two parts of the url (folder and filename)
            # do this to prevent special characters (such as ö) of causing problems down the line
            filename = urllib.parse.unquote(str(Path(*url.parts[-2:])))
            save_to_path(r.content, self.spiders_dir / filename)
        except Exception as e:
            self.logger.error(f"Caught an exception while processing {str(url)}\n{e}")


if __name__ == '__main__':
    config = get_config()

    scraper = Scraper(config)
    scraper.download_subfolders(base_url + "docs/")
