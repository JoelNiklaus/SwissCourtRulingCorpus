import gc
import multiprocessing
import sys
from pathlib import Path
import glob
from time import sleep
from memory_profiler import profile
import os, psutil
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Date, ARRAY

from sqlalchemy.dialects.postgresql import insert
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
from tqdm import tqdm
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import configparser

from tqdm.contrib.concurrent import process_map

from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from root import ROOT_DIR
from scrc.utils.decorators import slack_alert
from scrc.utils.log_utils import get_logger

# import scrc.utils.monkey_patch  # prevent memory leak with pandas

# IMPORTANT: make sure you download these models first with: python -m spacy download de_dep_news_trf
import de_core_news_lg, fr_core_news_lg, it_core_news_lg

from scrc.utils.main_utils import get_file_gen
from scrc.utils.slack_util import post_message_to_slack


class VocabularyComputer(DatasetConstructorComponent):
    """
    Runs the entire spacy pipeline for each text and saves it into the MongoDB.
    This brings the advantage, that we have the heavy computation done in advance,
    and can then use the spacy objects directly in our analysis.

    Here is a very good resource to reduce memory consumption: https://pythonspeed.com/memory/
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.lang_dir = None
        self.spacy_vocab = None

    def run_pipeline(self):
        self.logger.info("Started computing vocabulary")

        for lang in self.languages:
            self.logger.info(f"Started processing language {lang}")
            self.lang_dir = self.create_dir(self.spacy_subdir, lang)  # output dir

            engine = self.get_engine()

            meta = MetaData()
            lang_chambers_table = Table(  # an aggregate table for storing chamber specific data like the vocabulary
                f"{lang}_chambers", meta,
                Column('chamber', String, primary_key=True),
                Column('vocabulary', ARRAY(String)),
            )
            meta.create_all(engine)

            chambers = self.query(f"SELECT DISTINCT chamber FROM {lang}").chamber.to_list()
            processed_file_path = self.data_dir / f"{lang}_chambers_vocabularied.txt"
            spider_list, message = self.compute_remaining_parts(processed_file_path, chambers)
            self.logger.info(message)

            if chambers:
                self.spacy_vocab = self.load_vocab(self.lang_dir)

            for chamber in chambers:
                chamber_vocab = self.compute_vocabulary(engine, chamber, lang)
                with engine.connect() as conn:
                    stmt = insert(lang_chambers_table).values(chamber=chamber, vocabulary=chamber_vocab)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=[lang_chambers_table.c.chamber],
                        set_=dict(vocabulary=chamber_vocab)
                    )
                    conn.execute(stmt)
                self.mark_as_processed(processed_file_path, chamber)

            self.logger.info(f"Finished processing language {lang}")

        self.logger.info("Finished computing vocabulary")

    def compute_vocabulary(self, engine, chamber, lang):
        self.logger.info(f"Processing chamber {chamber}")

        dfs = self.select(engine, lang, columns='id', where=f"chamber='{chamber}'")  # stream dfs from the db
        total_vocab = set()
        for df in dfs:
            ids = df.id.to_list()
            self.logger.info("Loading spacy docs")  # load
            docs = [Doc(self.spacy_vocab).from_disk(self.lang_dir / (str(id) + ".spacy"), exclude=['tensor'])
                    for id in ids]
            self.logger.info("Computing the vocabs")  # map
            vocabs = tqdm(map(self.create_vocab_for_doc, docs), total=len(ids))
            self.logger.info("Aggregating vocabs by taking the union")  # reduce
            total_vocab |= set.union(*vocabs)  # take the union of all individual vocabs
        gc.collect()
        sleep(2)  # sleep(2) is required to allow measurement of the garbage collector

        self.logger.info(f"Chamber {chamber} has a vocabulary of {len(total_vocab)} words")

        return total_vocab

    def create_vocab_for_doc(self, doc: spacy.tokens.Doc) -> set:
        """
        take lemma without underscore for faster computation (int instead of str)
        take casefold of lemma to remove capital letters and ß
        """
        return set([token.lemma_.casefold()
                    for token in doc
                    if (not token.is_stop and not token.pos_ in ['NUM', 'PUNCT', 'SYM', 'X'])])


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    vocabulary_computer = VocabularyComputer(config)
    vocabulary_computer.run_pipeline()
