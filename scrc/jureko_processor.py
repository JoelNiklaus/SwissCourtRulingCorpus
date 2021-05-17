import configparser
import glob
from pathlib import Path

import spacy
from tqdm import tqdm

from tei_reader import TeiReader

from sqlalchemy import MetaData, Table, Column, Integer, String, Date, JSON

import pandas as pd

from root import ROOT_DIR
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.log_utils import get_logger


class JurekoProcessor(DatasetConstructorComponent):

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.types = {'decision', 'statute'}

    def process(self):
        engine = self.get_engine(self.db_jureko)
        self.extract_to_db(engine)

        disable_pipes = ['senter', 'ner', 'attribute_ruler', 'textcat']
        nlp = spacy.load('de_core_news_lg', disable=disable_pipes)
        nlp.max_length = 3000000

        self.logger.info("Running spacy pipeline")
        processed_file_path = self.jureko_subdir / f"types_spacied.txt"
        types, message = self.compute_remaining_parts(processed_file_path, self.types)
        self.logger.info(message)
        for type in types:
            type_dir = self.create_dir(self.jureko_spacy_subdir, type)
            self.run_spacy_pipe(engine, type, type_dir, "", nlp, self.logger)
            self.mark_as_processed(processed_file_path, type)

        self.logger.info("Computing counters")
        processed_file_path = self.jureko_subdir / f"types_counted.txt"
        types, message = self.compute_remaining_parts(processed_file_path, self.types)
        self.logger.info(message)
        for type in types:
            type_dir = self.jureko_spacy_subdir / type
            spacy_vocab = self.load_vocab(type_dir)
            self.compute_counters(engine, type, "", spacy_vocab, type_dir, self.logger)
            self.mark_as_processed(processed_file_path, type)

        self.compute_total_aggregate(engine, self.types, "type", self.jureko_subdir, self.logger)

    def extract_to_db(self, engine):
        reader = TeiReader()

        file_names = [Path(file_path).name for file_path in glob.glob(str(self.tei_subdir / "*.txt"))]
        processed_file_path = self.jureko_subdir / f"files_extracted.txt"
        files, message = self.compute_remaining_parts(processed_file_path, file_names)
        self.logger.info(message)
        entries = {'text': [], 'table': [], 'title': [], 'date': [], 'file_number': [], }
        i = 0
        for file in tqdm(files, total=len(file_names)):
            self.process_file(entries, self.jureko_subdir / file, reader)
            self.mark_as_processed(processed_file_path, file)
            self.save_to_db(engine, entries, i)
            i += 1

    def save_to_db(self, engine, entries, i):
        if i % self.chunksize == 0:  # save to db in chunks so we don't overload RAM
            df = pd.DataFrame.from_dict(entries)
            for type in set(df.type.to_list()):
                self.types.add(type)  # update types list
                self.create_type_table(engine, type)
                type_df = df[df.type.str.contains(type, na=False)]  # select only decisions by table
                if len(type_df.index) > 0:
                    type_df.to_sql(type, engine, if_exists="append", index=False)
            entries = {'text': [], 'table': [], 'title': [], 'date': [], 'file_number': [], }

    def process_file(self, entries, file, reader):
        type_key = 'fileDesc::sourceDesc::biblStruct::table'
        title_key = 'fileDesc::titleStmt::title'
        date_key = 'fileDesc::sourceDesc::biblStruct::analytic'
        # the first one of these should be the Aktenzeichen
        file_number_key = 'fileDesc::sourceDesc::biblStruct::analytic::idno'
        corpora = reader.read_file(file)  # or read_string
        entries['text'].append(corpora.text)
        entries['table'].append(self.get_attribute(corpora, type_key))
        entries['title'].append(self.get_attribute(corpora, title_key))
        entries['file_number'].append(self.get_attribute(corpora, file_number_key))
        date = self.get_attribute(corpora, date_key)
        if date == 'NoDate':
            date = None
        entries['date'].append(date)

    def get_attribute(self, corpora, key):
        attributes = list(corpora.documents)[0].attributes
        for a in attributes:
            if a.key == key:
                return a.text

    def create_type_table(self, engine, type):
        meta = MetaData()
        type_table = Table(
            type, meta,
            Column('id', Integer, primary_key=True),
            Column('text', String),
            Column('table', String),
            Column('title', String),
            Column('date', Date),
            Column('file_number', String),
            Column('num_tokens', Integer),
            Column('lemma_counter', JSON),
            Column('pos_counter', JSON),
            Column('tag_counter', JSON),
        )
        meta.create_all(engine)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    jureko_processor = JurekoProcessor(config)
    jureko_processor.process()
