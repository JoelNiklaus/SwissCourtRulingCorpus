import json

import glob
from pathlib import Path

import pymongo
import subprocess
from sqlalchemy import create_engine
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
import pandas as pd
import configparser

from root import ROOT_DIR
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_file_gen


class DBImporter(DatasetConstructorComponent):
    """
    Imports the csv data to a MongoDB and creates indexes on the fields most used for querying
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.indexes = json.loads(config['mongodb']['indexes'])

        self.engine = create_engine(
            "postgresql+psycopg2://postgres:postgres@localhost:5432/scrc"
        )

    def process_sql_using_pandas(self, lang):
        with self.engine.connect().execution_options(stream_results=True) as conn:
            query = f"SELECT * FROM {lang}"
            for chunk_dataframe in pd.read_sql(query, conn, chunksize=1000):
                print(f"Got dataframe w/{len(chunk_dataframe)} rows")

    def import_data(self):
        for lang in self.languages:
            self.import_chambers(lang)
            self.create_indexes(lang)

    def import_chambers(self, lang):
        self.logger.info(f'Importing chamber files for {lang} into DB')

        dir = self.split_subdir / lang
        chamber_list = [Path(chamber).name for chamber in glob.glob(f"{str(dir)}/*")]
        for chamber in chamber_list:
            self.logger.info(f"Importing {chamber} to DB")
            file_gen = get_file_gen(dir / chamber)
            if file_gen:
                for chunk, df in file_gen:
                    df = df.drop(['html_clean', 'pdf_clean', 'html_raw', 'html_clean'], axis='columns', errors='ignore')
                    df['text'] = df['text'].replace('\x00', '')
                    self.logger.info(f"Inserting {len(df.index)} decisions from chunk {chunk}")
                    df.to_sql(lang, self.engine, if_exists="append", index=False)

    def create_indexes(self, lang):
        self.logger.info(f"Creating indexes for {lang}")
        with self.engine.connect() as conn:
            for index in self.indexes:
                self.logger.info(f"Creating index for column {index} in table {lang}")
                conn.execute(f"CREATE INDEX {lang}_{index} ON {lang}({index})")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    db_importer = DBImporter(config)
    db_importer.import_data()
