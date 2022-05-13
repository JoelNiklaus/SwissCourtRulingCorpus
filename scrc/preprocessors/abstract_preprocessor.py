from datetime import datetime
import gc
import importlib
import json
import multiprocessing
import os
from collections import Counter, Sized
from pathlib import Path
import glob
from time import sleep
from typing import Optional

import spacy
from spacy.lang.de import German
from spacy.lang.fr import French
from spacy.lang.it import Italian
from spacy.tokens import Doc
from spacy.vocab import Vocab
from tqdm import tqdm
from transformers import AutoTokenizer

from root import ROOT_DIR
import pandas as pd

from sqlalchemy.sql.expression import bindparam
from sqlalchemy import create_engine, MetaData, Table, Column, String, JSON
from sqlalchemy.dialects.postgresql import insert

from stopwordsiso import stopwords

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)  # set to None to impose no limit


class AbstractPreprocessor:
    """
    Serves as an abstract base class for preprocessors.
    Contains fields with directories and database access information
    and methods for interacting with the database and saving progress.
    """

    def __init__(self, config: dict):
        self.languages = json.loads(config['general']['languages'])
        self.chunksize = int(config['general']['chunksize'])
        self.ignore_cache = config['general']['ignore_cache'].lower() == 'true'

        self.data_dir = self.create_dir(ROOT_DIR, config['dir']['data_dir'])
        self.progress_dir = self.create_dir(
            self.data_dir, config['dir']['progress_dir'])
        self.spider_specific_dir = self.create_dir(
            ROOT_DIR, config['dir']['spider_specific_dir'])

        self.spiders_dir = self.create_dir(
            self.data_dir, config['dir']['spiders_subdir'])
        self.spacy_subdir = self.create_dir(
            self.data_dir, config['dir']['spacy_subdir'])
        self.datasets_subdir = self.create_dir(
            self.data_dir, config['dir']['datasets_subdir'])
        self.tmp_subdir = self.create_dir(
            self.data_dir, config['dir']['tmp_subdir'])

        self.corpora_subdir = self.create_dir(
            self.data_dir, config['dir']['corpora_subdir'])
        self.slc_subdir = self.create_dir(
            self.corpora_subdir, config['dir']['slc_subdir'])
        self.slc_spacy_subdir = self.create_dir(
            self.slc_subdir, config['dir']['spacy_subdir'])
        self.jureko_subdir = self.create_dir(
            self.corpora_subdir, config['dir']['jureko_subdir'])
        self.jureko_spacy_subdir = self.create_dir(
            self.jureko_subdir, config['dir']['spacy_subdir'])
        self.wikipedia_subdir = self.create_dir(
            self.corpora_subdir, config['dir']['wikipedia_subdir'])
        self.wikipedia_spacy_subdir = self.create_dir(
            self.wikipedia_subdir, config['dir']['spacy_subdir'])
        self.spider_specific_dir = self.create_dir(
            ROOT_DIR, config['dir']['spider_specific_dir'])
        self.output_dir = self.create_dir(
            self.data_dir, config['dir']['output_subdir'])

        self.ip = config['postgres']['ip']
        self.port = config['postgres']['port']
        self.user = config['postgres']['user']
        self.password = config['postgres']['password']
        self.db_scrc = config['postgres']['db_scrc']
        self.db_jureko = config['postgres']['db_jureko']
        self.db_slc = config['postgres']['db_slc']
        self.db_wikipedia = config['postgres']['db_wikipedia']

        self.indexes = json.loads(config['postgres']['indexes'])

        self.num_cpus = multiprocessing.cpu_count()

        self.stopwords = stopwords(self.languages)
        # this should be filtered out by PUNCT pos tag already, but sometimes they are misclassified
        self.stopwords |= {' ', '.', '!', '?'}

        self.counter_types = ['counter_lemma', 'counter_pos', 'counter_tag']

    @staticmethod
    def create_dir(parent_dir: Path, dir_name: str) -> Path:
        dir = parent_dir / dir_name
        # create folder if it does not exist yet
        dir.mkdir(parents=True, exist_ok=True)
        return dir

    def load_functions(self, config, type):
        """loads the cleaning functions used for html files"""
        function_file = self.spider_specific_dir / \
            config['files'][type]  # mainly used for html courts
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
        spider_list = [Path(spider).stem for spider in glob.glob(
            f"{str(self.spiders_dir)}/*")]
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
    def _check_write_privilege(engine) -> bool:
        return AbstractPreprocessor.query(engine, 'SELECT current_user')['current_user'][0] != 'readonly'

    def add_column(self, engine, table, col_name, data_type) -> None:
        """
        Adds a column to an existing table
        :param engine:
        :param table:
        :param col_name:
        :param data_type:
        :return:
        """
        if not self._check_write_privilege(engine):
            return
        with engine.connect() as conn:
            query = f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col_name} {data_type}"
            conn.execute(query)

    @staticmethod
    def select(engine, table, columns="*", where=None, order_by=None, chunksize=1000):
        """
        This is the utility function to stream entries from the database.

        :param engine:          the db engine to work upon
        :param table:           the table (language) to select
        :param columns:         the columns to retrieve (comma separated list)
        :param where:           an sql WHERE clause to filter by certain column values
        :param order_by:        an sql ORDER BY clause to order the output
        :param chunksize:       the number of rows to retrieve per chunk
        :return:                a generator of pd.DataFrame
        """
        with engine.connect().execution_options(stream_results=True) as conn:
            query = f"SELECT {columns} FROM {table}"
            if where:
                query += " WHERE " + where
            if order_by:
                query += " ORDER BY " + order_by
            for chunk_df in pd.read_sql(query, conn, chunksize=chunksize):
                yield chunk_df

    @staticmethod
    def update(engine, df: pd.DataFrame, table: str, columns: list, output_dir: Path, filename: Optional[str] = None, index_name: Optional[str] = None):
        """
        Updates the given columns in a table with the data provided by the df
        :param engine:              the db engine to work upon
        :param df:                  the df providing the data for the update
        :param table:               the table to be updated
        :param columns:             the columns to be updated
        :param output_dir:          the output directory if output stored as files instead of the database (readonly users)
        :param filename:            the filename to be used for the output (if stored as files)
        :return:
        """

        if not AbstractPreprocessor._check_write_privilege(engine):
            path = ''
            if filename is None:
                AbstractPreprocessor.create_dir(output_dir, os.getlogin())
                path = Path.joinpath(output_dir, os.getlogin(
                ), datetime.now().isoformat() + '.json')
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                path = Path.joinpath(output_dir, filename)
            with path.open("a") as f:
                df.to_json(f)
            return

        with engine.connect() as conn:
            t = Table(table, MetaData(), autoload_with=engine)  # get the table
            # only update these cols, id needs to be there for the where clause
            if not index_name:
                df = df[columns + ['id']]
                # cannot use the same name as the col name
                df = df.rename(columns={'id': 'b_id'})
            else:
                df = df[columns + [index_name]]
                if index_name == 'decision_id':
                    df = df.astype({'decision_id': str})
                # cannot use the same name as the col name
                df = df.rename(columns={index_name: 'b_id'})
            # updates all columns which are present in the df
            query = t.update().where(t.c.get(index_name or 'id') ==
                                     bindparam('b_id')).values()
            conn.execute(query, df.to_dict('records'))  # bulk update

    @staticmethod
    def load_vocab(spacy_dir) -> Vocab:
        vocab_path = spacy_dir / f"_vocab.spacy"
        if vocab_path.exists():
            return Vocab().from_disk(str(vocab_path), exclude=['vectors'])

    @staticmethod
    def save_vocab(vocab, spacy_dir) -> None:
        vocab.to_disk(spacy_dir / f"_vocab.spacy", exclude=['vectors'])

    def run_nlp_pipe(self, engine, table, spacy_dir, where, nlp, bert_tokenizer, logger):
        """
        Runs the spacy pipe on the table provided and saves the docs into the given folder
        :param engine:      the engine with the db connection
        :param table:       where to get the data from and to save it to
        :param spacy_dir:   where to save the docs obtained
        :param where:       how to select the dfs
        :param nlp:         used for creating the docs
        :param bert_tokenizer: used for computing the number of bert tokens if present
        :param logger:      custom logger for info output
        :return:
        """
        dfs = self.select(engine, table, columns='id, text',
                          where=where)  # stream dfs from the db
        for df in dfs:
            # reorder the df so that we get the text first and the id after
            df = df[['text', 'id']]
            # convert df to list of tuples
            tuples = list(df.itertuples(index=False))
            # batch_size = max(int(len(texts) / self.num_cpus), 1) # a high batch_size can lead to lots of allocated memory
            docs = tqdm(nlp.pipe(tuples, n_process=-1, batch_size=1,
                        as_tuples=True), total=len(tuples))
            num_tokens = []
            logger.info("Saving spacy docs to disk")
            for doc, id in docs:
                path = spacy_dir / (str(id) + ".spacy")
                # this makes the space on the disk much smaller!
                doc.to_disk(path, exclude=['tensor'])
                num_tokens.append(len(doc))
            df['num_tokens_spacy'] = num_tokens

            if bert_tokenizer:
                df['num_tokens_bert'] = [
                    len(input_id) for input_id in bert_tokenizer(df['text'].tolist()).input_ids]

            columns = ['num_tokens_spacy', 'num_tokens_bert']
            logger.info("Saving num_tokens_spacy and num_tokens_bert to db")
            self.update(engine, df, table, columns, self.output_dir)

            self.save_vocab(nlp.vocab, spacy_dir)

            gc.collect()
            # sleep(2) is required to allow measurement of the garbage collector
            sleep(2)

    def get_tokenizers(self, lang):
        os.environ['TOKENIZERS_PARALLELISM'] = "True"
        if lang == 'de':
            spacy = German()
            bert = "deepset/gbert-base"
        elif lang == 'fr':
            spacy = French()
            bert = "camembert/camembert-base-ccnet"
        elif lang == 'it':
            spacy = Italian()
            bert = "dbmdz/bert-base-italian-cased"
        else:
            raise ValueError(
                f"Please choose one of the following languages: {self.languages}")
        return spacy.tokenizer, AutoTokenizer.from_pretrained(bert)

    @staticmethod
    def insert_counter(engine, table, level, level_instance, counter_type, counter):
        """Inserts a counter into an aggregate table"""
        with engine.connect() as conn:
            values = {level: level_instance, counter_type: counter}
            stmt = insert(table).values(values)
            stmt = stmt.on_conflict_do_update(
                index_elements=[table.c[level]],
                set_={counter_type: counter}
            )
            conn.execute(stmt)

    def compute_total_aggregate(self, engine, tables, tables_name, base_dir, logger):
        logger.info("Aggregating counters")
        agg_table = self.create_aggregate_table(engine, f"agg", tables_name)
        processed_file_path = base_dir / f"{tables_name}s_aggregated.txt"
        tables_remaining, message = self.compute_remaining_parts(
            processed_file_path, tables)
        logger.info(message)
        for table in tables_remaining:
            for counter_type in self.counter_types:
                aggregate_counter = self.compute_aggregate_counter(
                    engine, table, "", counter_type, logger)
                self.insert_counter(
                    engine, agg_table, tables_name, table, counter_type, aggregate_counter)
            self.mark_as_processed(processed_file_path, table)

    def compute_aggregate_counter(self, engine, table: str, where: str, counter_type, logger) -> dict:
        """Computes an aggregate counter for the dfs queried by the parameters"""
        logger.info(f"Computing aggregate counter for {table}")
        dfs = self.select(engine, table, columns=counter_type,
                          where=where)  # stream dfs from the db
        aggregate_counter = Counter()
        for df in dfs:
            for counter in tqdm(df[counter_type].to_list()):
                aggregate_counter += Counter(counter)
        return dict(aggregate_counter)

    def create_aggregate_table(self, engine, table_name, primary_key):
        """Creates an aggregate table for a given level for storing the counter"""
        meta = MetaData()
        table = Table(  # an aggregate table for storing level specific data like the vocabulary
            table_name, meta,
            Column(primary_key, String, primary_key=True),
            Column('counter_lemma', JSON),
            Column('counter_pos', JSON),
            Column('counter_tag', JSON),
        )
        meta.create_all(engine)
        return table

    def compute_counters(self, engine, table, where, spacy_vocab, spacy_dir, logger):
        """Computes the counter for each of the decisions in a given chamber and language"""
        dfs = self.select(engine, table, columns='id',
                          where=where)  # stream dfs from the db
        for df in dfs:
            ids = df.id.to_list()
            logger.info(f"Loading {len(ids)} spacy docs")  # load
            docs = [Doc(spacy_vocab).from_disk(spacy_dir / (str(id) + ".spacy"), exclude=['tensor'])
                    for id in ids]
            for counter_type in self.counter_types:
                logger.info(
                    f"Computing the counters for type {counter_type}")  # map
                counter_type_list = [counter_type] * len(docs)
                df[counter_type] = tqdm(
                    map(self.create_counter_for_doc, docs, counter_type_list), total=len(ids))
            self.update(engine, df, table, self.counter_types,
                        self.output_dir)  # save
        gc.collect()
        sleep(2)  # sleep(2) is required to allow measurement of the garbage collector

    def create_counter_for_doc(self, doc: spacy.tokens.Doc, counter_type, filter_stops=False) -> dict:
        """
        take lemma without underscore for faster computation (int instead of str)
        """
        if counter_type == 'counter_lemma':
            # take casefold of lemma to remove capital letters and ß
            lemmas = [token.lemma_.casefold() for token in doc if token.pos_ not in [
                'NUM', 'PUNCT', 'SYM', 'X']]
            # only take alphanumeric lemmas to filter out digits
            lemmas = [lemma for lemma in lemmas if lemma.isalpha()]
            if filter_stops:  # don't do this at the moment because it can still be done later if needed
                # filter out stopwords (spacy is_stop filtering does not work well)
                lemmas = [
                    lemma for lemma in lemmas if lemma not in self.stopwords]
            return dict(Counter(lemmas))
        elif counter_type == 'counter_pos':
            poses = [token.pos_ for token in doc]
            return dict(Counter(poses))
        elif counter_type == 'counter_tag':
            tags = [token.tag_ for token in doc]
            return dict(Counter(tags))
        else:
            raise ValueError(
                f"You chose counter_type {counter_type}. Please choose one of {self.counter_types}.")
