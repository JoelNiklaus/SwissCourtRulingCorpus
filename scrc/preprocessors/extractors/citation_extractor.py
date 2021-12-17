from typing import Optional, Union
import configparser
import bs4
import pandas as pd
from sqlalchemy.engine.base import Engine
from sqlalchemy.sql.expression import text
from sqlalchemy.sql.schema import MetaData, Table

from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor
from root import ROOT_DIR
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config


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

    def select_df(self, engine: str, spider: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        return self.select(engine, 'file LEFT JOIN decision on decision.file_id = file.file_id LEFT JOIN language ON language.language_id = decision.language_id', f"decision_id, iso_code as language, html_raw, pdf_raw, '{spider}' as spider", where=f"file.file_id IN (SELECT file_id from decision WHERE chamber_id IN (SELECT chamber_id FROM chamber WHERE spider_id IN (SELECT spider_id FROM spider WHERE spider.name = '{spider}')))", chunksize=self.chunksize)
    
    
    def save_data_to_database(self, df: pd.DataFrame, engine: Engine):
        key_to_id = {
            'ruling': 1,
            'law': 2,
            'commentary': 3
        }
        
        for idx, row in df.iterrows():
            with engine.connect() as conn:
                t = Table('citation', MetaData(), autoload_with=engine)
                stmt = t.delete().where(text(f"decision_id in ({','.join([chr(39)+str(item)+chr(39) for item in df['decision_id'].tolist()])})"))
                engine.execute(stmt)
                for k in row['citations'].keys():
                    citation_type_id = key_to_id[k]
                    for citation in row['citations'][k]:
                        stmt = t.insert().values([{"decision_id": str(row['decision_id']), "citation_type_id": citation_type_id, "url": citation.get("url"), "text": citation["text"]}])
                        engine.execute(stmt)

if __name__ == '__main__':
    config = get_config()

    citation_extractor = CitationExtractor(config)
    citation_extractor.start()
