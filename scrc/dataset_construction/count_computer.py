import gc
from collections import Counter
from time import sleep
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Date, ARRAY, PickleType, JSON
from sqlalchemy.dialects.postgresql import insert
import spacy
from spacy.tokens import Doc
from tqdm import tqdm
import configparser
import stopwordsiso as stopwords

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
        self.stopwords = stopwords.stopwords(self.languages)
        # this should be filtered out by PUNCT pos tag already, but sometimes they are misclassified
        self.stopwords |= {' ', '.', '!', '?'}

    def run_pipeline(self):
        self.logger.info("Started computing counts")

        for lang in self.languages:
            self.logger.info(f"Started processing language {lang}")
            self.lang_dir = self.create_dir(self.spacy_subdir, lang)  # output dir

            engine = self.get_engine()
            self.compute_counts_for_individual_decisions(engine, lang)

            self.compute_aggregates(engine, lang)

            self.logger.info(f"Finished processing language {lang}")

        self.logger.info("Finished computing counts")

    def compute_counts_for_individual_decisions(self, engine, lang):
        self.add_column(engine, lang, col_name='counter', data_type='jsonb')  # add new column for the rank_order

        chambers = self.get_level_instances(lang, 'chamber')
        processed_file_path = self.data_dir / f"{lang}_chambers_counted.txt"
        chambers, message = self.compute_remaining_parts(processed_file_path, chambers)
        self.logger.info(message)

        if chambers:
            self.logger.info("Computing the counters for individual decisions")
            self.spacy_vocab = self.load_vocab(self.lang_dir)

            for chamber in chambers:
                self.compute_counters(engine, chamber, lang)
                self.mark_as_processed(processed_file_path, chamber)

    def compute_aggregates(self, engine, lang):
        meta = MetaData()
        lang_chambers_table = self.create_aggregate_table(lang, meta, 'chamber')
        lang_courts_table = self.create_aggregate_table(lang, meta, 'court')
        lang_cantons_table = self.create_aggregate_table(lang, meta, 'canton')
        meta.create_all(engine)

        self.logger.info("Computing the aggregate counters for the chambers")
        chambers = self.get_level_instances(lang, 'chamber')
        for chamber in chambers:
            self.compute_counters(engine, chamber, lang)
            chamber_counter = self.compute_aggregate_counter(engine, lang, f"chamber='{chamber}'")
            self.insert_counter(engine, lang_chambers_table, 'chamber', chamber, chamber_counter)

        self.logger.info("Computing the aggregate counters for the courts")
        courts = self.get_level_instances(lang, 'court')
        for court in courts:
            court_counter = self.compute_aggregate_counter(engine, f"{lang}_chambers", f"chamber LIKE '{court}_%'")
            self.insert_counter(engine, lang_courts_table, 'court', court, court_counter)

        self.logger.info("Computing the aggregate counters for the cantons")
        cantons = self.get_level_instances(lang, 'canton')
        for canton in cantons:
            canton_counter = self.compute_aggregate_counter(engine, f"{lang}_courts", f"court LIKE '{canton}_%'")
            self.insert_counter(engine, lang_cantons_table, 'canton', canton, canton_counter)

    def get_level_instances(self, lang, level):
        return self.query(f"SELECT DISTINCT {level} FROM {lang}")[level].to_list()

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

    def compute_counters(self, engine, chamber, lang):
        """Computes the counter for each of the decisions in a given chamber and language"""
        self.logger.info(f"Processing chamber {chamber}")

        dfs = self.select(engine, lang, columns='id', where=f"chamber='{chamber}'")  # stream dfs from the db
        for df in dfs:
            ids = df.id.to_list()
            self.logger.info(f"Loading {len(ids)} spacy docs")  # load
            docs = [Doc(self.spacy_vocab).from_disk(self.lang_dir / (str(id) + ".spacy"), exclude=['tensor'])
                    for id in ids]
            self.logger.info("Computing the counters")  # map
            df['counter'] = tqdm(map(self.create_counter_for_doc, docs), total=len(ids))
            self.update(engine, df, lang, ['counter'])  # save
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


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    count_computer = CountComputer(config)
    count_computer.run_pipeline()
