from abc import ABC, abstractmethod
from scrc.utils.log_utils import get_logger
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
import pandas as pd

from typing import Any

class Extractor(ABC, DatasetConstructorComponent):

    
    loggerInfo = {
        'start': 'Started extracting', 
        'finished': 'Finished extracting', 
        'start_spider': 'Started extracting for spider', 
        'finish_spider': 'Finished extracting for spider', 
        'saving': 'Saving extracted part',
        'processing_one': 'Extracting court decision'
        }
    processed_file_path = ''

    def __init__(self, config: dict, function_name: str, col_name: str):
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.processing_functions = self.load_functions(config, function_name)
        self.logger.debug(self.processing_functions)
        self.col_name = col_name

    def start(self):
        self.logger.info(self.loggerInfo['start'])
        spider_list, message = self.getProcessedSpiders()
        self.logger.info(message)

        engine = self.get_engine(self.db_scrc)
        self.addColumns(engine)

        self.startSpiderLoop(spider_list, engine)

        

    def getProcessedSpiders(self):
        self.processed_file_path = self.data_dir / "spiders_judgement_extracted.txt"
        spider_list, message = self.compute_remaining_spiders(self.processed_file_path)
        return spider_list, message

    def addColumns(self, engine):
        for lang in self.languages:
            self.add_column(engine, lang, col_name=self.col_name, data_type='jsonb')

    def startSpiderLoop(self, spider_list, engine):
        for spider in spider_list:
            if hasattr(self.processing_functions, spider):
                self.processOneSpider(engine, spider)
            else:
                self.logger.debug(f"There are no special functions for spider {spider}. "
                                  f"Not performing any judgement extracting.")
            self.mark_as_processed(self.processed_file_path, spider)

    def processOneSpider(self, engine, spider):
        self.logger.info(self.loggerInfo['start_spider'])
        

        for lang in self.languages:
            where = self.getDatabaseSelectionString(spider, lang)
            
            chunksize = 1000
            dfs = self.select(engine, lang, where=where, chunksize=chunksize)  # stream dfs from the db
            for df in dfs:
                df = df.apply(self.processOneDFRow, axis='columns')
                self.update(engine, df, lang, [self.col_name])
                self.logProgress(chunksize)
            
            self.logCoverage(engine, lang)

        self.logger.info(f"Finished lower-court-extracting {spider}")

    def processOneDFRow(self, series):
        """Processes one row of a raw df"""
        self.logger.debug(f"{self.loggerInfo['processing_one']} {series['file_name']}")
        namespace = series[['date', 'language', 'html_url', 'id']].to_dict()
        data = self.getRequiredData(series)
        if not data:
            return
        series[self.col_name] = self.processOneEntry(series['spider'], data, namespace)

    def processOneEntry(self, spider: str, data: Any, namespace: dict):
        extracting_functions = getattr(self.processing_functions, spider)
        try:
            return extracting_functions(data, namespace)  # invoke function with data and namespace
        except ValueError as e:
            self.logger.warning(e)
            return None  # just ignore the error for now. It would need much more rules to prevent this.

    def startProgress(self, engine, lang):
        self.processedAmount = 0
        self.totalToProcess = self.coverage_getTotal(engine, lang)

    def logProgress(self, chunksize):
        self.processedAmount = max(self.processedAmount + chunksize, self.totalToProcess)
        self.logger.info(f"{self.loggerInfo['saving']} ({self.processedAmount}/{self.totalToProcess})")

    def logCoverage(self, engine, lang):
        successful_attempts = self.coverage_getSuccessful(engine, lang)
        self.logger.info(
            f"{self.loggerInfo['finish_spider']} in {lang} with "
            f"{successful_attempts} / {self.totalToProcess} "
            f"({successful_attempts / self.totalToProcess:.2%}) "
            "working"
        )
    
    @abstractmethod
    def getRequiredData(self, series) -> Any:
        pass

    @abstractmethod
    def coverage_getTotal(self, engine, lang) -> int:
        """
        Returns total amount of valid entries to be processed by extractor
        """
        pass

    @abstractmethod
    def getDatabaseSelectionString(self, spider, lang) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        pass

    @abstractmethod
    def coverage_getSuccessful(self, engine, lang) -> int:
        """Returns the total entries that got processed successfully"""
        pass