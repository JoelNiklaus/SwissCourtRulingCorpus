import gc
import json
import multiprocessing
from collections import Counter, Sized
from pathlib import Path

import glob
from time import sleep

import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
from tqdm import tqdm

from root import ROOT_DIR
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy import MetaData, Table
from sqlalchemy.sql.expression import bindparam
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Date, ARRAY, PickleType, JSON
from sqlalchemy.dialects.postgresql import insert

import stopwordsiso as stopwords


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
        self.spacy_subdir = self.create_dir(self.data_dir, config['dir']['spacy_subdir'])
        self.kaggle_subdir = self.create_dir(self.data_dir, config['dir']['kaggle_subdir'])

        self.corpora_subdir = self.create_dir(self.data_dir, config['dir']['corpora_subdir'])
        self.jureko_subdir = self.create_dir(self.corpora_subdir, config['dir']['jureko_subdir'])
        self.tei_subdir = self.create_dir(self.jureko_subdir, config['dir']['tei_subdir'])
        self.jureko_spacy_subdir = self.create_dir(self.jureko_subdir, config['dir']['spacy_subdir'])

        self.ip = config['postgres']['ip']
        self.port = config['postgres']['port']
        self.user = config['postgres']['user']
        self.password = config['postgres']['password']
        self.db_scrc = config['postgres']['db_scrc']
        self.db_jureko = config['postgres']['db_jureko']

        self.indexes = json.loads(config['postgres']['indexes'])

        self.num_cpus = multiprocessing.cpu_count()

        self.stopwords = stopwords.stopwords(self.languages)
        # this should be filtered out by PUNCT pos tag already, but sometimes they are misclassified
        self.stopwords |= {' ', '.', '!', '?'}

    @staticmethod
    def create_dir(parent_dir: Path, dir_name: str) -> Path:
        dir = parent_dir / dir_name
        dir.mkdir(parents=True, exist_ok=True)  # create folder if it does not exist yet
        return dir

    def load_functions(self, config, type):
        """loads the cleaning functions used for html files"""
        function_file = ROOT_DIR / config['files'][type]  # mainly used for html courts
        spec = importlib.util.spec_from_file_location(type, function_file)
        functions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(functions)
        return functions

    @staticmethod
    def mark_as_processed(processed_file_path: Path, part: str) -> None:
        with processed_file_path.open("a") as f:
            f.write(part + "\n")

    def compute_remaining_spiders(self, processed_file_path: Path):
        """This can be used to save progress in between runs in case something fails"""
        spider_list = [Path(spider).stem for spider in glob.glob(f"{str(self.spiders_dir)}/*")]
        return self.compute_remaining_parts(processed_file_path, spider_list)

    @staticmethod
    def compute_remaining_parts(processed_file_path: Path, entire: Sized):
        if not processed_file_path.exists():
            processed_file_path.touch()
        processed = processed_file_path.read_text().strip().split("\n")
        left_to_process = set(entire) - set(processed)
        message = f"Still {len(left_to_process)} of {len(entire)} part(s) remaining to process: {left_to_process}"
        return left_to_process, message

    def get_engine(self, db, echo=False):
        return create_engine(
            f"postgresql+psycopg2://{self.user}:{self.password}@{self.ip}:{self.port}/{db}",
            echo=echo  # good for debugging
        )

    @staticmethod
    def query(engine, query_str) -> pd.DataFrame:
        """
        Simple method to query anything from the database
        :param engine:      the engine used for the db connection
        :param query_str:   the sql statement to send
        :return:
        """
        with engine.connect() as conn:
            return pd.read_sql(query_str, conn)

    @staticmethod
    def add_column(engine, table, col_name, data_type) -> None:
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

    @staticmethod
    def select(engine, table, columns="*", where=None, chunksize=1000):
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

    @staticmethod
    def update(engine, df: pd.DataFrame, table: str, columns: list):
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
            columns.append('id')  # id needs to be there for the where clause
            df = df[columns]  # only update these cols
            df = df.rename(columns={'id': 'b_id'})  # cannot use the same name as the col name
            # updates all columns which are present in the df
            query = t.update().where(t.c.id == bindparam('b_id')).values()
            conn.execute(query, df.to_dict('records'))  # bulk update

    @staticmethod
    def load_vocab(spacy_dir) -> Vocab:
        vocab_path = spacy_dir / f"_vocab.spacy"
        if vocab_path.exists():
            return Vocab().from_disk(str(vocab_path), exclude=['vectors'])

    @staticmethod
    def save_vocab(vocab, spacy_dir) -> None:
        vocab.to_disk(spacy_dir / f"_vocab.spacy", exclude=['vectors'])

    def run_spacy_pipe(self, engine, table, spacy_dir, where, nlp, logger):
        """
        Runs the spacy pipe on the table provided and saves the docs into the given folder
        :param engine:      the engine with the db connection
        :param table:       where to get the data from and to save it to
        :param spacy_dir:   where to save the docs obtained
        :param where:       how to select the dfs
        :param nlp:         used for creating the docs
        :param logger:      custom logger for info output
        :return:
        """
        dfs = self.select(engine, table, columns='id, text', where=where)  # stream dfs from the db
        for df in dfs:
            df = df[['text', 'id']]  # reorder the df so that we get the text first and the id after
            tuples = list(df.itertuples(index=False))  # convert df to list of tuples
            # batch_size = max(int(len(texts) / self.num_cpus), 1) # a high batch_size can lead to lots of allocated memory
            docs = tqdm(nlp.pipe(tuples, n_process=-1, batch_size=1, as_tuples=True), total=len(tuples))
            num_tokens = []
            logger.info("Saving spacy docs to disk")
            for doc, id in docs:
                path = spacy_dir / (str(id) + ".spacy")
                doc.to_disk(path, exclude=['tensor'])
                num_tokens.append(len(doc))
            df['num_tokens'] = num_tokens
            columns = ['num_tokens']
            logger.info("Saving num_tokens to db")
            self.update(engine, df, table, columns)

            self.save_vocab(nlp.vocab, spacy_dir)

            gc.collect()
            sleep(2)  # sleep(2) is required to allow measurement of the garbage collector


    @staticmethod
    def insert_counter(engine, table, level, level_instance, counter):
        """Inserts a counter into an aggregate table"""
        with engine.connect() as conn:
            values = {level: level_instance, "counter": counter}
            stmt = insert(table).values(values)
            stmt = stmt.on_conflict_do_update(
                index_elements=[table.c[level]],
                set_=dict(counter=counter)
            )
            conn.execute(stmt)

    def compute_aggregate_counter(self, engine, table: str, where: str) -> dict:
        """Computes an aggregate counter for the dfs queried by the parameters"""
        dfs = self.select(engine, table, columns='counter', where=where)  # stream dfs from the db
        aggregate_counter = Counter()
        for df in dfs:
            for counter in df.counter.to_list():
                aggregate_counter += Counter(counter)
        return dict(aggregate_counter)

    def create_aggregate_table(self, engine, table, primary_key):
        """Creates an aggregate table for a given level for storing the counter"""
        meta = MetaData()
        lang_level_table = Table(  # an aggregate table for storing level specific data like the vocabulary
            table, meta,
            Column(primary_key, String, primary_key=True),
            Column('counter', JSON),
        )
        meta.create_all(engine)
        return lang_level_table

    def compute_counters(self, engine, table, where, spacy_vocab, spacy_dir, logger):
        """Computes the counter for each of the decisions in a given chamber and language"""
        dfs = self.select(engine, table, columns='id', where=where)  # stream dfs from the db
        for df in dfs:
            ids = df.id.to_list()
            logger.info(f"Loading {len(ids)} spacy docs")  # load
            docs = [Doc(spacy_vocab).from_disk(spacy_dir / (str(id) + ".spacy"), exclude=['tensor'])
                    for id in ids]
            logger.info("Computing the counters")  # map
            df['counter'] = tqdm(map(self.create_counter_for_doc, docs), total=len(ids))
            self.update(engine, df, table, ['counter'])  # save
        gc.collect()
        sleep(2)  # sleep(2) is required to allow measurement of the garbage collector

    def create_counter_for_doc(self, doc: spacy.tokens.Doc) -> dict:
        """
        take lemma without underscore for faster computation (int instead of str)
        take casefold of lemma to remove capital letters and ß
        """
        lemmas = [token.lemma_.casefold() for token in doc if token.pos_ not in ['NUM', 'PUNCT', 'SYM', 'X']]
        # filter out stopwords (spacy is_stop filtering does not work well)
        lemmas = [lemma for lemma in lemmas if lemma not in self.stopwords and lemma.isalpha()]
        return dict(Counter(lemmas))
