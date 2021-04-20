import json
import multiprocessing
from pathlib import Path

import glob

from root import ROOT_DIR
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy import MetaData, Table
from sqlalchemy.sql.expression import bindparam

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)  # set to None to impose no limit


class DatasetConstructorComponent:
    """
    Extracts the textual and meta information from the court rulings files and saves it in csv files for each court
    and in one for all courts combined
    """

    def __init__(self, config: dict):
        self.languages = json.loads(config['general']['languages'])
        self.chunksize = int(config['general']['chunksize'])

        self.data_dir = self.create_dir(ROOT_DIR, config['dir']['data_dir'])
        self.spiders_dir = self.create_dir(self.data_dir, config['dir']['spiders_subdir'])
        self.raw_subdir = self.create_dir(self.data_dir, config['dir']['raw_subdir'])
        self.clean_subdir = self.create_dir(self.data_dir, config['dir']['clean_subdir'])
        self.split_subdir = self.create_dir(self.data_dir, config['dir']['split_subdir'])
        self.spacy_subdir = self.create_dir(self.data_dir, config['dir']['spacy_subdir'])
        self.kaggle_subdir = self.create_dir(self.data_dir, config['dir']['kaggle_subdir'])

        self.ip = config['postgres']['ip']
        self.port = config['postgres']['port']
        self.database = config['postgres']['database']
        self.user = config['postgres']['user']
        self.password = config['postgres']['password']

        self.indexes = json.loads(config['postgres']['indexes'])

        self.num_cpus = multiprocessing.cpu_count()

    def create_dir(self, parent_dir, dir_name):
        dir = parent_dir / dir_name
        dir.mkdir(parents=True, exist_ok=True)  # create folder if it does not exist yet
        return dir

    def mark_as_processed(self, processed_file_path, spider):
        with processed_file_path.open("a") as f:
            f.write(spider + "\n")

    def compute_remaining_spiders(self, processed_file_path):
        """This can be used to save progress in between runs in case something fails"""
        spider_list = [Path(spider).stem for spider in glob.glob(f"{str(self.spiders_dir)}/*")]
        if not processed_file_path.exists():
            processed_file_path.touch()
        spiders_processed = processed_file_path.read_text().strip().split("\n")
        spiders_not_yet_processed = set(spider_list) - set(spiders_processed)
        message = f"Still {len(spiders_not_yet_processed)} of {len(spider_list)} spider(s) remaining to process: {spiders_not_yet_processed}"
        return spiders_not_yet_processed, message

    def get_engine(self):
        return create_engine(
            f"postgresql+psycopg2://{self.user}:{self.password}@{self.ip}:{self.port}/{self.database}",
            # echo=True # good for debugging
        )

    def add_column(self, engine, table, col_name, data_type):
        """
        Adds a column to an existing table
        :param engine:
        :param table:
        :param col_name:
        :param data_type:
        :return:
        """
        with engine.connect() as conn:
            query = f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col_name} {data_type}"
            conn.execute(query)

    def select(self, engine, table, columns="*", where=None, chunksize=1000):
        """
        This is the utility function to stream entries from the database.

        :param engine:      the db engine to work upon
        :param table:       the table (language) to select
        :param columns:     the columns to retrieve (comma separated list)
        :param where:       an sql WHERE clause to filter by certain collumn values
        :param chunksize:   the number of rows to retrieve per chunk
        :return:            a generator of pd.DataFrame
        """
        with engine.connect().execution_options(stream_results=True) as conn:
            query = f"SELECT {columns} FROM {table}"
            if where:
                query += " WHERE " + where
            for chunk_df in pd.read_sql(query, conn, chunksize=chunksize):
                yield chunk_df

    def update(self, engine, df: pd.DataFrame, table: str, columns: list):
        """
        Updates the given columns in a table with the data provided by the df
        :param engine:              the db engine to work upon
        :param df:                  the df providing the data for the update
        :param table:               the table to be updated
        :param columns:             the columns to be updated
        :return:
        """
        with engine.connect() as conn:
            t = Table(table, MetaData(), autoload_with=engine)  # get the table
            columns.append('uuid')  # uuid needs to be there for the where clause
            df = df[columns]  # only update these cols
            df = df.rename(columns={'uuid': 'b_uuid'})  # cannot use the same name as the col name
            # updates all columns which are present in the df
            query = t.update().where(t.c.uuid == bindparam('b_uuid')).values()
            conn.execute(query, df.to_dict('records'))  # bulk update
