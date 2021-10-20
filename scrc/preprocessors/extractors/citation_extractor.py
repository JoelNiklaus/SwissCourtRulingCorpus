from typing import Optional, Union
import configparser
import bs4
import pandas as pd

from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor
from root import ROOT_DIR
from scrc.utils.log_utils import get_logger


class CitationExtractor(AbstractExtractor):
    """
    Extract citations from the html documents. This represents the citation extraction task.
    """

    def __init__(self, config: dict):
        super().__init__(config, function_name='citation_extracting_functions', col_name='citations')
        self.logger = get_logger(__name__)
        self.processed_file_path = self.progress_dir / "spiders_citation_extracted.txt"
        self.logger_info = {
            'start': 'Started extracting citations',
            'finished': 'Finished extracting citations',
            'start_spider': 'Started extracting citations for spider',
            'finish_spider': 'Finished extracting citations for spider',
            'saving': 'Saving chunk of citations',
            'processing_one': 'Extracting citations from',
            'no_functions': 'Not extracting citations.'
        }

    def get_required_data(self, series: pd.DataFrame) -> Union[bs4.BeautifulSoup, str, None]:
        """Returns the data required by the processing functions"""
        html_raw = series['html_raw']
        if pd.notna(html_raw) and html_raw not in [None, '']:
            # Parses the html string with bs4 and returns the body content
            return bs4.BeautifulSoup(html_raw, "html.parser").find('body')
        pdf_raw = series['pdf_raw']
        if pd.notna(pdf_raw) and pdf_raw not in [None, '']:
            return pdf_raw
        return None

    def get_database_selection_string(self, spider: str, lang: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        return f"spider='{spider}'"


if __name__ == '__main__':
    config = configparser.ConfigParser()
    # this stops working when the script is called from the src directory!
    config.read(ROOT_DIR / 'config.ini')

    citation_extractor = CitationExtractor(config)
    citation_extractor.start()
