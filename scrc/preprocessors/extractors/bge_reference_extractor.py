from __future__ import annotations
from typing import Any, TYPE_CHECKING, Union
import bs4
import pandas as pd
from sqlalchemy.engine.base import Engine
from root import ROOT_DIR
from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor
from scrc.utils.main_utils import get_config
from scrc.utils.sql_select_utils import join_decision_and_language_on_parameter, \
    where_decisionid_in_list, where_string_spider

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame


class BgeReferenceExtractor(AbstractExtractor):
    """
    Extracts the reference to a bger from bge header section. This is part of the criticality prediction task.
    """

    def __init__(self, config: dict):
        super().__init__(config, function_name='bge_reference_extracting_functions', col_name='bge_reference')
        self.processed_file_path = self.progress_dir / "bge_reference_extracted.txt"
        self.logger_info = {
            'start': 'Started extracting the bge references',
            'finished': 'Finished extracting the bge references',
            'start_spider': 'Started extracting the bge references for spider',
            'finish_spider': 'Finished extracting the bge references for spider',
            'saving': 'Saving chunk of bge references',
            'processing_one': 'Extracting the bge references from',
            'no_functions': 'Not extracting the bge references'
        }

    def get_database_selection_string(self, spider: str, lang: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        return f"spider='{spider}' AND header IS NOT NULL AND header <> ''"

    def get_required_data(self, series: pd.DataFrame) -> Union[bs4.BeautifulSoup, str, None]:
        """Returns the data required by the processing functions"""
        html_raw = series['html_raw']
        if pd.notna(html_raw) and html_raw not in [None, '']:
            # Parses the html string with bs4 and returns the body content
            return bs4.BeautifulSoup(html_raw, "html.parser")
        pdf_raw = series['pdf_raw']
        if pd.notna(pdf_raw) and pdf_raw not in [None, '']:
            return pdf_raw
        return None

    def select_df(self, engine: str, spider: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        only_given_decision_ids_string = f" AND {where_decisionid_in_list(self.decision_ids)}" if self.decision_ids is not None else ""
        return self.select(engine, f"file {join_decision_and_language_on_parameter('file_id', 'file.file_id')}",
                           f"decision_id, iso_code as language, html_raw, pdf_raw, '{spider}' as spider",
                           where=f"file.file_id IN {where_string_spider('file_id', spider)} {only_given_decision_ids_string}",
                           chunksize=self.chunksize)


    def save_data_to_database(self, df: pd.DataFrame, engine: Engine):
        """Instead of saving data into database, references get written into a text file"""
        self.logger.info("save data in progress")
        processed_file_path = ROOT_DIR / 'data' / 'progress' / "bge_references_found.txt"
        for _, row in df.iterrows():
            # only add new content to textfile not overwriting
            with processed_file_path.open("a") as f:
                if 'bge_reference' in row and row['bge_reference'] != 'no reference found':
                    bge_reference = str(row['bge_reference'])
                    f.write(f"{bge_reference}\n")


    def check_condition_before_process(self, spider: str, data: Any, namespace: dict) -> bool:
        """Override if data has to conform to a certain condition before processing.
        e.g. data is required to be present for analysis"""
        return bool(data)


if __name__ == '__main__':
    config = get_config()
    bge_reference_extractor = BgeReferenceExtractor(config)
    # delete bge_reference_found.txt file before letting it run.
    # remove CH_BGE from processed spiders
    bge_reference_extractor.start()
