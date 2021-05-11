import configparser
from typing import Optional

from root import ROOT_DIR
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.log_utils import get_logger


class JudgementExtractor(DatasetConstructorComponent):
    """
    Extracts the judgements from the rulings section
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.judgement_extracting_functions = self.load_functions(config, 'judgement_extracting_functions')
        self.logger.debug(self.judgement_extracting_functions)

    def extract_judgements(self):
        """splits sections of all the raw court rulings with the defined functions"""
        self.logger.info("Started judgement-extracting raw court rulings")

        processed_file_path = self.data_dir / "spiders_judgement_extracted.txt"
        spider_list, message = self.compute_remaining_spiders(processed_file_path)
        self.logger.info(message)

        engine = self.get_engine(self.db_scrc)
        for lang in self.languages:
            self.add_column(engine, lang, col_name='judgement', data_type='text')

        # clean raw texts
        for spider in spider_list:
            if hasattr(self.judgement_extracting_functions, spider):
                self.section_split_spider(engine, spider)
            else:
                self.logger.debug(f"There are no special functions for spider {spider}. "
                                  f"Not performing any section splitting.")
            self.mark_as_processed(processed_file_path, spider)

        self.logger.info("Finished judgement-extracting raw court rulings")

    def section_split_spider(self, engine, spider):
        """Extracts judgements for one spider"""
        self.logger.info(f"Started judgement-extracting {spider}")

        for lang in self.languages:
            dfs = self.select(engine, lang, where=f"spider='{spider}'")  # stream dfs from the db
            for df in dfs:
                df = df.apply(self.section_split_df_row, axis='columns')
                self.logger.info("Saving cleaned text and split sections to db")
                self.update(engine, df, lang, ['judgement'])

        self.logger.info(f"Finished judgement-extracting {spider}")

    def section_split_df_row(self, series):
        """Processes one row of a raw df"""
        self.logger.debug(f"Judgement-extracting court decision {series['file_name']}")
        namespace = series[['date', 'language', 'html_url']].to_dict()
        series['judgement'] = self.extract_judgement_with_functions(series['spider'], series['rulings'], namespace)

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
