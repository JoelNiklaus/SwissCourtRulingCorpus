from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, Set, TYPE_CHECKING, Tuple
import pandas as pd

from root import ROOT_DIR
from scrc.utils.log_utils import get_logger
from scrc.preprocessing.abstract_preprocessor import AbstractPreprocessor

if TYPE_CHECKING:
    from sqlalchemy.engine.base import Engine


class AbstractExtractor(ABC, AbstractPreprocessor):
    """Abstract Base Class used by Extractors to unify their behaviour"""

    logger_info = {
        "start": "Started extracting",
        "finished": "Finished extracting",
        "start_spider": "Started extracting for spider",
        "finish_spider": "Finished extracting for spider",
        "saving": "Saving extracted part",
        "processing_one": "Extracting court decision",
        "no_functions": "Not processing",
    }
    processed_file_path = None

    @abstractmethod
    def get_required_data(self, series: pd.DataFrame) -> Any:
        """Returns the data required by the processing functions"""

    @abstractmethod
    def get_database_selection_string(self, spider: str, lang: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""

    def check_condition_before_process(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing.
        e.g. data is required to be present for analysis"""
        return True

    def __init__(self, config: dict, function_name: str, col_name: str):
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.processing_functions = self.load_functions(config, function_name)
        self.logger.debug(self.processing_functions)
        self.col_name = col_name
        self.processed_amount = 0
        self.total_to_process = -1
        self.spider_specific_dir = self.create_dir(ROOT_DIR, config['dir']['spider_specific_dir'])

    def start(self):
        self.logger.info(self.logger_info["start"])
        spider_list, message = self.get_processed_spiders()
        self.logger.info(message)

        engine = self.get_engine(self.db_scrc)
        self.add_columns(engine)

        self.start_spider_loop(spider_list, engine)

        self.logger.info(self.logger_info["finished"])

    def get_processed_spiders(self) -> Tuple[Set, str]:
        spider_list, message = self.compute_remaining_spiders(self.processed_file_path)
        return spider_list, message

    def add_columns(self, engine: Engine):
        for lang in self.languages:
            self.add_column(engine, lang, col_name=self.col_name, data_type="jsonb")

    def start_spider_loop(self, spider_list: Set, engine: Engine):
        for spider in spider_list:
            if hasattr(self.processing_functions, spider):
                self.process_one_spider(engine, spider)
            else:
                self.logger.debug(
                    f"There are no special functions for spider {spider}. "
                    f"{self.logger_info['no_functions']}"
                )
            self.mark_as_processed(self.processed_file_path, spider)

    def process_one_spider(self, engine: Engine, spider: str):
        self.logger.info(self.logger_info["start_spider"] + " " + spider)

        for lang in self.languages:
            where = self.get_database_selection_string(spider, lang)
            self.start_progress(engine, spider, lang)
            # stream dfs from the db
            dfs = self.select(engine, lang, where=where,
                              chunksize=self.chunksize)
            for df in dfs:
                df = df.apply(self.process_one_df_row, axis="columns")
                self.update(engine, df, lang, [self.col_name])
                self.log_progress(self.chunksize)

            self.log_coverage(engine, spider, lang)

        self.logger.info(f"{self.logger_info['finish_spider']} {spider}")

    def process_one_df_row(self, series: pd.DataFrame) -> pd.DataFrame:
        """Processes one row of a raw df"""
        self.logger.debug(
            f"{self.logger_info['processing_one']} {series['file_name']}")
        namespace = series[["date", "language", "html_url", "id"]].to_dict()
        data = self.get_required_data(series)
        assert data
        series[self.col_name] = self.call_processing_function(
            series["spider"], data, namespace
        )
        return series

    def call_processing_function(self, spider: str, data: Any, namespace: dict) -> Optional[Any]:
        if not self.check_condition_before_process(spider, data, namespace):
            return None
        extracting_functions = getattr(self.processing_functions, spider)
        try:
            # invoke function with data and namespace
            return extracting_functions(data, namespace)
        except ValueError as e:
            self.logger.warning(e)
            # just ignore the error for now. It would need much more rules to prevent this.
            return None

    def coverage_get_total(self, engine: Engine, spider: str, lang: str) -> int:
        """
        Returns total amount of valid entries to be processed by extractor
        """
        sql_string = f"SELECT count(*) FROM {lang} WHERE {self.get_database_selection_string(spider, lang)}"
        return pd.read_sql(sql_string, engine.connect())["count"][0]

    def coverage_get_successful(self, engine: Engine, spider: str, lang: str) -> int:
        """Returns the total entries that got processed successfully"""
        query = (
            f"SELECT count({self.col_name}) FROM {lang} WHERE "
            f"{self.get_database_selection_string(spider, lang)} AND {self.col_name} <> 'null'"
        )
        return pd.read_sql(query, engine.connect())["count"][0]

    def start_progress(self, engine: Engine, spider: str, lang: str):
        self.processed_amount = 0
        self.total_to_process = self.coverage_get_total(engine, spider, lang)
        self.logger.info(f"Total: {self.total_to_process}")

    def log_progress(self, chunksize: int):
        self.processed_amount = min(
            self.processed_amount + chunksize, self.total_to_process)
        self.logger.info(f"{self.logger_info['saving']} ({self.processed_amount}/{self.total_to_process})")

    def log_coverage(self, engine: Engine, spider: str, lang: str):
        successful_attempts = self.coverage_get_successful(engine, spider, lang)
        self.logger.info(
            f"{self.logger_info['finish_spider']} in {lang} with "
            f"{successful_attempts} / {self.total_to_process} "
            f"({successful_attempts / self.total_to_process:.2%}) "
            "working"
        )
