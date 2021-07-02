import configparser
from typing import Optional

from root import ROOT_DIR
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.log_utils import get_logger
import pandas as pd


class JudgementExtractor(DatasetConstructorComponent):
    """
    Extracts the judgements from the rulings section
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.judgement_extracting_functions = self.load_functions(config, 'judgement_extracting_functions')
        self.logger.debug(self.judgement_extracting_functions)

        self.col_name = "judgements"

    def extract_judgements(self):
        """splits sections of all the raw court rulings with the defined functions"""
        self.logger.info("Started judgement-extracting raw court rulings")

        processed_file_path = self.data_dir / "spiders_judgement_extracted.txt"
        spider_list, message = self.compute_remaining_spiders(processed_file_path)
        self.logger.info(message)

        engine = self.get_engine(self.db_scrc)
        for lang in self.languages:
            self.add_column(engine, lang, col_name=self.col_name, data_type='jsonb')

        # extracting judgements from rulings
        for spider in spider_list:
            if hasattr(self.judgement_extracting_functions, spider):
                self.judgement_extract_spider(engine, spider)
            else:
                self.logger.debug(f"There are no special functions for spider {spider}. "
                                  f"Not performing any judgement extracting.")
            self.mark_as_processed(processed_file_path, spider)

        self.logger.info("Finished judgement-extracting raw court rulings")

    def judgement_extract_spider(self, engine, spider):
        """Extracts judgements for one spider"""
        self.logger.info(f"Started judgement-extracting {spider}")

        for lang in self.languages:
            where = f"spider='{spider}' AND rulings IS NOT NULL AND rulings <> ''"
            count = pd.read_sql(f"SELECT count(*) FROM {lang} WHERE {where}", engine.connect())['count'][0]
            dfs = self.select(engine, lang, where=where)  # stream dfs from the db
            for df in dfs:
                df = df.apply(self.judgement_extract_df_row, axis='columns')
                self.logger.info("Saving extracted judgements to db")
                self.update(engine, df, lang, [self.col_name])

            query = f"SELECT count({self.col_name}) FROM {lang} WHERE {where} AND {self.col_name} <> 'null'"
            successful_attempts = pd.read_sql(query, engine.connect())['count'][0]
            self.logger.info(f"Finished judgement-extracting  {spider} in {lang} with "
                             f"{successful_attempts} / {count} ({successful_attempts / count:.2%}) working")

        self.logger.info(f"Finished judgement-extracting {spider}")

    def judgement_extract_df_row(self, series):
        """Processes one row of a raw df"""
        self.logger.debug(f"Judgement-extracting court decision {series['file_name']}")
        namespace = series[['date', 'language', 'html_url']].to_dict()
        series[self.col_name] = self.extract_judgement_with_functions(series['spider'], series['rulings'], namespace)

        return series

    def extract_judgement_with_functions(self, spider: str, rulings: str, namespace: dict) -> Optional:
        if not rulings:  # the rulings could not be extracted in every case
            return None
        judgement_extracting_functions = getattr(self.judgement_extracting_functions, spider)
        try:
            # invoke function with text and namespace
            return judgement_extracting_functions(rulings, namespace)
        except ValueError as e:
            self.logger.warning(e)
            return None  # just ignore the error for now. It would need much more rules to prevent this.


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    judgement_extractor = JudgementExtractor(config)
    judgement_extractor.extract_judgements()
