import configparser
from typing import Optional

from root import ROOT_DIR
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.log_utils import get_logger
import pandas as pd


class LowerCourtExtractor(DatasetConstructorComponent):
    """
    Extracts the lower courts from the header section
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.lower_court_extracting_functions = self.load_functions(config, 'lower_court_extracting_functions')
        self.logger.debug(self.lower_court_extracting_functions)

        self.col_name = "lower_court"

    def extract_lower_courts(self):
        self.logger.info("Started extracting lower courts from headers")

        processed_file_path = self.data_dir / "spiders_lower_court_extracted.txt"
        spider_list, message = self.compute_remaining_spiders(processed_file_path)
        self.logger.info(message)

        engine = self.get_engine(self.db_scrc)
        for lang in self.languages:
            self.add_column(engine, lang, col_name=self.col_name, data_type='jsonb')

        # extracting judgements from rulings
        for spider in spider_list:
            if hasattr(self.lower_court_extracting_functions, spider):
                self.lower_court_extract_spider(engine, spider)
            else:
                self.logger.debug(f"There are no special functions for spider {spider}. "
                                  f"Not performing any lower court extracting.")
            self.mark_as_processed(processed_file_path, spider)

        self.logger.info("Finished lower-court-extracting raw court rulings")

    def lower_court_extract_spider(self, engine, spider):
        """Extracts lower courts for one spider"""
        self.logger.info(f"Started lower-court-extracting {spider}")
        for lang in self.languages:
            counter = 0
            where = f"spider='{spider}' AND header IS NOT NULL AND header <> ''"
            count = pd.read_sql(f"SELECT count(*) FROM {lang} WHERE {where}", engine.connect())['count'][0]
            dfs = self.select(engine, lang, where=where)  # stream dfs from the db
            for df in dfs:
                df = df.apply(self.lower_court_extract_df_row, axis='columns')
                self.logger.info("Saving extracted lower courts to db")
                self.update(engine, df, lang, [self.col_name])
                counter = counter + 1000
                print(counter, count, sep=" / ")
            query = f"SELECT count({self.col_name}) FROM {lang} WHERE {where} AND {self.col_name} <> 'null'"
            successful_attempts = pd.read_sql(query, engine.connect())['count'][0]
            self.logger.info(f"Finished lower-court-extracting  {spider} in {lang} with "
                            f"{successful_attempts} / {count} ({successful_attempts / count:.2%}) working")

        self.logger.info(f"Finished lower-court-extracting {spider}")

    def lower_court_extract_df_row(self, series):
        """Processes one row of a raw df"""
        self.logger.debug(f"Lower-court-extracting court decision {series['file_name']}")
        namespace = series[['date', 'language', 'html_url']].to_dict()
        series[self.col_name] = self.extract_lower_court_with_functions(series['spider'], series['header'], namespace)

        return series

    def extract_lower_court_with_functions(self, spider: str, header: str, namespace: dict) -> Optional:
        if not header:  # the rulings could not be extracted in every case
            return None
        lower_court_extracting_functions = getattr(self.lower_court_extracting_functions, spider)
        try:
            # invoke function with text and namespace
            return lower_court_extracting_functions(header, namespace, self.get_engine(self.db_scrc))
        except ValueError as e:
            self.logger.warning(e)
            return None  # just ignore the error for now. It would need much more rules to prevent this.


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    lower_court_extractor = LowerCourtExtractor(config)
    lower_court_extractor.extract_lower_courts()
