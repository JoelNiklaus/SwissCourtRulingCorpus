import gc
from collections import Counter
from time import sleep
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Date, ARRAY, PickleType, JSON
from sqlalchemy.dialects.postgresql import insert
import spacy
from spacy.tokens import Doc
from tqdm import tqdm
import configparser

from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from root import ROOT_DIR
from scrc.utils.decorators import slack_alert
from scrc.utils.log_utils import get_logger

# import scrc.utils.monkey_patch  # prevent memory leak with pandas

"""
TODO: Here we have the problem that comparing corpora of unequal length is very difficult.
=> 
1. From cleaned text: sklearn CountVectorizer/HashingVectorizer to get bow 
2. take top k (1000-10000) frequent words 
3. take the intersection of these two lists and divide by k to get overlap

Maybe use two bow representations to calculate tf-idf and then compare top k words of the two corpora
"""


class CountComputer(DatasetConstructorComponent):
    """
    Computes the lemma counts for each decision and saves it in a special column 'counter'.
    In a second step, it computes the aggregate counts for the chamber, court and canton level
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.lang_dir = None
        self.spacy_vocab = None

    @slack_alert
    def run_pipeline(self):
        self.logger.info("Started computing counts")

        engine = self.get_engine(self.db_scrc)
        for lang in self.languages:
            self.logger.info(f"Started processing language {lang}")
            self.lang_dir = self.spacy_subdir / lang

            self.compute_counts_for_individual_decisions(engine, lang)
            self.compute_aggregates(engine, lang)

            self.logger.info(f"Finished processing language {lang}")

        self.logger.info("Finished computing counts")

    def compute_counts_for_individual_decisions(self, engine, lang):
        self.add_column(engine, lang, col_name='counter', data_type='jsonb')  # add new column for the rank_order

        chambers = self.get_level_instances(engine, lang, 'chamber')
        processed_file_path = self.data_dir / f"{lang}_chambers_counted.txt"
        chambers, message = self.compute_remaining_parts(processed_file_path, chambers)
        self.logger.info(message)

        if chambers:
            self.logger.info("Computing the counters for individual decisions")
            self.spacy_vocab = self.load_vocab(self.lang_dir)

            for chamber in chambers:
                self.logger.info(f"Processing chamber {chamber}")
                self.compute_counters(engine, lang, f"chamber='{chamber}'", self.spacy_vocab, self.lang_dir,
                                      self.logger)
                self.mark_as_processed(processed_file_path, chamber)

    def compute_aggregates(self, engine, lang):
        compile_where = lambda level_instance: f"chamber='{level_instance}'"
        self.compute_aggregate_for_level(engine, lang, 'chamber', lang, compile_where)

        compile_where = lambda level_instance: f"chamber LIKE '{level_instance}_%'"
        self.compute_aggregate_for_level(engine, lang, 'court', f"{lang}_chambers", compile_where)

        compile_where = lambda level_instance: f"court LIKE '{level_instance}_%'"
        self.compute_aggregate_for_level(engine, lang, 'canton', f"{lang}_courts", compile_where)

    def compute_aggregate_for_level(self, engine, lang, level, table, compile_where):
        meta = MetaData()
        lang_level_table = self.create_aggregate_table(lang, meta, level)
        meta.create_all(engine)

        self.logger.info(f"Computing the aggregate counters for the {level}s")
        level_instances = self.get_level_instances(engine, lang, level)
        processed_file_path = self.data_dir / f"{lang}_{level}s_aggregated.txt"
        level_instances, message = self.compute_remaining_parts(processed_file_path, level_instances)
        self.logger.info(message)

        for level_instance in level_instances:
            self.logger.info(f"Processing {level} {level_instance}")
            where = compile_where(level_instance)
            aggregate_counter = self.compute_aggregate_counter(engine, table, where)
            self.insert_counter(engine, lang_level_table, level, level_instance, aggregate_counter)
            self.mark_as_processed(processed_file_path, level_instance)

    def get_level_instances(self, engine, lang, level):
        return self.query(engine, f"SELECT DISTINCT {level} FROM {lang}")[level].to_list()

    def insert_counter(self, engine, lang_level_table, level, level_instance, counter):
        """Inserts a counter into an aggregate table"""
        with engine.connect() as conn:
            values = {level: level_instance, "counter": counter}
            stmt = insert(lang_level_table).values(values)
            stmt = stmt.on_conflict_do_update(
                index_elements=[lang_level_table.c[level]],
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

    def create_aggregate_table(self, lang, meta, level):
        """Creates an aggregate table for a given level for storing the counter"""
        lang_level_table = Table(  # an aggregate table for storing level specific data like the vocabulary
            f"{lang}_{level}s", meta,
            Column(level, String, primary_key=True),
            Column('counter', JSON),
        )
        return lang_level_table


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    count_computer = CountComputer(config)
    count_computer.run_pipeline()
