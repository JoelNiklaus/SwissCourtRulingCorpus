from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Set, TYPE_CHECKING, Tuple
import pandas as pd
from root import ROOT_DIR

from scrc.enums.language import Language
from scrc.utils.log_utils import get_logger
from scrc.preprocessors.abstract_preprocessor import AbstractPreprocessor

if TYPE_CHECKING:
    from sqlalchemy.engine.base import Engine


class AbstractExtractor(ABC, AbstractPreprocessor):
    """Abstract Base Class used by Extractors to unify their behaviour"""

    # Basic logging information made to overwrite in subclasses
    logger_info = {
        "start": "Started extracting",  # Used at the start of the extraction
        "finished": "Finished extracting",  # Used at the end of the extraction
        "start_spider": "Started extracting for spider",  # Used when a new spider is starting extraction
        "finish_spider": "Finished extracting for spider",  # Used when a spider is finished extracting
        "saving": "Saving extracted part",  # Used when saving extracted data to database
        "processing_one": "Extracting court decision",  # Used to log when a single instance is being processed
        "no_functions": "Not processing",
        # Used when the extraction is skipped because there are no functions for the spider
    }
    processed_file_path = None

    @abstractmethod
    def get_required_data(self, series: pd.DataFrame) -> Any:
        """Returns the data required by the processing functions"""

    @abstractmethod
    def save_data_to_database(self, series: pd.DataFrame, engine: Engine):
        """Splits the data into their respective parts and saves them to the table"""
        
    @abstractmethod
    def get_coverage(self, engine: Engine, spider: str):
        """Logs the coverage"""

    @abstractmethod
    def select_df(self, engine: Engine, spider: str):
        """Selects the dataframe from the database"""

    def check_condition_before_process(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing.
        e.g. data is required to be present for analysis"""
        return True

    def __init__(self, config: dict, function_name: str, col_name: str, col_type: str = 'jsonb'):
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.processing_functions = self.load_functions(config, function_name)
        self.logger.debug(self.processing_functions)
        self.col_name = col_name
        self.col_type = col_type
        self.processed_amount = 0
        self.total_to_process = -1
        self.spider_specific_dir = self.create_dir(
            ROOT_DIR, config['dir']['spider_specific_dir'])

    def start(self, decision_ids: Optional[List] = None):
        """ Starting point for the extraction, calls the loop which processes each spider. Decision_ids can be specified to only process a subset of the data """
        self.logger.info(self.logger_info["start"])
        if self.rebuild_entire_database:
            self.processed_file_path.unlink()  # Delete the progress file if it exists to start from scratch
        if self.process_new_files_only:
            self.decision_ids = decision_ids
        else:
            self.decision_ids = None  # means: process all decision ids
        spider_list, message = self.get_processed_spiders()
        self.logger.info(message)

        engine = self.get_engine(self.db_scrc)
        # self.add_columns(engine)

        self.start_spider_loop(spider_list, engine)

        self.logger.info(self.logger_info["finished"])

    def get_processed_spiders(self) -> Tuple[Set, str]:
        spider_list, message = self.compute_remaining_spiders(
            self.processed_file_path)
        return spider_list, message

    """  def add_columns(self, engine: Engine):
        for lang in self.languages:
            self.add_column(engine, lang, col_name=self.col_name, data_type=self.col_type)
    """

    def start_spider_loop(self, spider_list: Set, engine: Engine):
        for spider in spider_list:
            try:
                # if there is no special function for the spider the default gets used
                if not hasattr(self.processing_functions, spider):
                    self.logger.debug(f"Using default function for {spider}")
                self.process_one_spider(engine, spider)
                self.mark_as_processed(self.processed_file_path, spider)
            except Exception as e:
                self.logger.exception(
                    f"There are no special functions for spider {spider} and the default spider does not work. "
                    f"{self.logger_info['no_functions']}")

    def process_one_spider(self, engine: Engine, spider: str):
        self.logger.info(self.logger_info["start_spider"] + " " + spider)

        dfs = self.select_df(self.get_engine(self.db_scrc), spider)  # Get the data needed for the extraction
        # TODO make quick request to see if there are decisions at all: if not, skip lang so no confusing report is printed
        # self.start_progress(engine, spider)
        # stream dfs from the db
        # dfs = self.select(engine, lang, where=where, chunksize=self.chunksize)
        for df in dfs:  # For each chunk in the data: apply the extraction function and save the result
            df = df.apply(self.process_one_df_row, axis="columns")
            df.to_csv('test.csv')
            # if not df.empty:
                # self.save_data_to_database(df, self.get_engine(self.db_scrc))
                # self.logger.info(f'One chunk of {len(df.index)} decisions saved')
            # self.log_progress(self.chunksize)
        self.get_coverage(spider)
        self.logger.info(f"{self.logger_info['finish_spider']} {spider}")

    def process_one_df_row(self, series: pd.DataFrame) -> pd.DataFrame:
        """Processes one row of a raw df"""
        namespace = dict()
        # Add the data to the namespace object which is passed to the extraction function
        namespace['html_url'] = series.get('html_url')
        namespace['pdf_url'] = series.get('pdf_url')
        namespace['date'] = series.get('date')
        namespace['language'] = Language(series['language'])
        namespace['id'] = series['decision_id']

        if 'court_string' in series:
            namespace['court'] = series.get('court_string')
        data = self.get_required_data(series)
        if not data:
            return series
        series[self.col_name] = self.call_processing_function(
            series["spider"], data, namespace
        )
        return series

    def call_processing_function(self, spider: str, data: Any, namespace: dict) -> Optional[Any]:
        """Calls the processing function (named by the spider) and passes the data and the namespace as arguments."""
        if not self.check_condition_before_process(spider, data, namespace):
            return None
        try:
            # Get the function for the spider or the default function
            default = getattr(self.processing_functions, 'XX_SPIDER')
            extracting_functions = getattr(self.processing_functions, spider, default)
            # invoke function with data and namespace
            return extracting_functions(data, namespace)
        except ValueError as e:
            # ToDo: info: using default, if it does not work then warning
            # Try block für default spider
            self.logger.warning(e)

            # just ignore the error for now. It would need much more rules to prevent this.
            return None

    def coverage_get_total(self, engine: Engine, spider: str, table: str) -> int:
        """
        Returns total amount of valid entries to be processed by extractor
        """
        sql_string = f"SELECT count(*) FROM {table} WHERE {self.get_database_selection_string(spider, table)}"
        return pd.read_sql(sql_string, engine.connect())["count"][0]

    def coverage_get_successful(self, engine: Engine, spider: str, table: str) -> int:
        """Returns the total entries that got processed successfully"""
        query = (
            f"SELECT count({self.col_name}) FROM {table} WHERE "
            f"{self.get_database_selection_string(spider)} AND {self.col_name} <> 'null'"
        )
        return pd.read_sql(query, engine.connect())["count"][0]

    def start_progress(self, engine: Engine, spider: str, lang: str):
        self.processed_amount = 0
        self.total_to_process = self.coverage_get_total(engine, spider, lang)
        self.logger.info(f"Total: {self.total_to_process}")

    def log_progress(self, chunksize: int):
        self.processed_amount = min(
            self.processed_amount + chunksize, self.total_to_process)
        self.logger.info(
            f"{self.logger_info['saving']} ({self.processed_amount}/{self.total_to_process})")

    def log_coverage(self, engine: Engine, spider: str, table: str):
        successful_attempts = self.coverage_get_successful(
            engine, spider, table)
        self.logger.info(
            f"{self.logger_info['finish_spider']} with "
            f"{successful_attempts} / {self.total_to_process} "
            f"({successful_attempts / self.total_to_process:.2%}) "
            "working"
        )
