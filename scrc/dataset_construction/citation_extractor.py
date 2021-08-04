import configparser
from scrc.utils.abstract_extractor import AbstractExtractor
from typing import Optional, Any

import bs4
import pandas as pd

from root import ROOT_DIR


class CitationExtractor(AbstractExtractor):
    """
    Extract citations from the html documents
    """

    def __init__(self, config: dict):
        super().__init__(config, function_name='citation_extracting_functions', col_name='citations')

    def getRequiredData(self, series) -> Any:
        html_raw = series['html_raw']
        if pd.notna(html_raw) and html_raw not in [None, '']:
            # Parses the html string with bs4 and returns the body content
            return bs4.BeautifulSoup(html_raw, "html.parser").find('body')
        return None

    def getDatabaseSelectionString(self, spider, lang) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        return f"spider='{spider}'"


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    citation_extractor = CitationExtractor(config)
    citation_extractor.processed_file_path = citation_extractor.data_dir / "spiders_citation_extracted.txt"
    citation_extractor.loggerInfo = {
        'start': 'Started extracting citations', 
        'finished': 'Finished extracting citations', 
        'start_spider': 'Started extracting citations for spider', 
        'finish_spider': 'Finished extracting citations for spider', 
        'saving': 'Saving chunk of citations',
        'processing_one': 'Extracting citations from',
        'no_functions': 'Not extracting citations.'
        }
    citation_extractor.start()
