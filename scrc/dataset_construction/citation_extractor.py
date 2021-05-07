import configparser
from typing import Optional, Any

import bs4
import pandas as pd

from root import ROOT_DIR
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.decorators import slack_alert
from scrc.utils.log_utils import get_logger


class CitationExtractor(DatasetConstructorComponent):
    """
    Extract citations from the html documents
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.citation_extracting_functions = self.load_functions(config, 'citation_extracting_functions')
        self.logger.debug(self.citation_extracting_functions)

    @slack_alert
    def extract_citations(self):
        """extract citations from all the raw court rulings with the defined functions"""
        self.logger.info("Started citation-extracting raw court rulings")

        processed_file_path = self.data_dir / "spiders_citation_extracted.txt"
        spider_list, message = self.compute_remaining_spiders(processed_file_path)
        self.logger.info(message)

        engine = self.get_engine(self.db_scrc)
        for lang in self.languages:
            self.add_column(engine, lang, col_name='citations', data_type='jsonb')

        for spider in spider_list:
            if hasattr(self.citation_extracting_functions, spider):
                self.citation_extract_spider(engine, spider)
            else:
                self.logger.debug(f"There are no special functions for spider {spider}. "
                                  f"Not performing any citation extraction.")
            self.mark_as_processed(processed_file_path, spider)

        self.logger.info("Finished citation-extracting raw court rulings")

    def citation_extract_spider(self, engine, spider):
        """Extracts citations for one spider"""
        self.logger.info(f"Started citation-extracting {spider}")

        for lang in self.languages:
            dfs = self.select(engine, lang, where=f"spider='{spider}'")  # stream dfs from the db
            for df in dfs:
                df = df.apply(self.citation_extract_df_row, axis='columns')
                self.logger.info("Saving extracted citations to db")
                self.update(engine, df, lang, ['citations'])

        self.logger.info(f"Finished citation-extracting {spider}")

    def citation_extract_df_row(self, series):
        """Extracts citations of one row of a raw df"""
        self.logger.debug(f"Citation-extracting court decision {series['file_name']}")
        namespace = series[['date', 'language', 'html_url']].to_dict()
        spider = series['spider']

        html_raw = series['html_raw']
        if pd.notna(html_raw) and html_raw not in [None, '']:
            # Parses the html string with bs4 and returns the body content
            soup = bs4.BeautifulSoup(html_raw, "html.parser").find('body')
            assert soup

            series['citations'] = self.split_sections_with_functions(spider, soup, namespace)

        return series

    def split_sections_with_functions(self, spider: str, soup: Any, namespace: dict) -> Optional:
        """Extract citations with citation extracting functions"""
        # retrieve function by spider
        citation_extracting_functions = getattr(self.citation_extracting_functions, spider)
        try:
            return citation_extracting_functions(soup, namespace)  # invoke function with soup and namespace
        except ValueError as e:
            self.logger.warning(e)
            return None  # just ignore the error for now. It would need much more rules to prevent this.


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    citation_extractor = CitationExtractor(config)
    citation_extractor.extract_citations()
