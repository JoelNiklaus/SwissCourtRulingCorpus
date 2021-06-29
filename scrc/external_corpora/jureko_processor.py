import configparser

from tei_reader import TeiReader

from sqlalchemy import MetaData, Table, Column, Integer, String, Date, JSON

import pandas as pd

from root import ROOT_DIR
from scrc.external_corpora.external_corpus_processor import ExternalCorpusProcessor
from scrc.utils.log_utils import get_logger


class JurekoProcessor(ExternalCorpusProcessor):

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.subdir = self.jureko_subdir
        self.spacy_subdir = self.jureko_spacy_subdir
        self.db = self.db_jureko
        self.tables_name = "type"
        self.tables = {'decision', 'statute'}
        self.entries_template = {'text': [], 'table': [], 'title': [], 'date': [], 'file_number': []}
        self.glob_str = "tei/*.txt"

    def save_to_db(self, engine, entries, i):
        if i % self.chunksize == 0:  # save to db in chunks so we don't overload RAM
            df = pd.DataFrame.from_dict(entries)
            for table in set(df.table.to_list()):
                self.create_type_table(engine, table)
                table_df = df[df.table.str.contains(table, na=False)]  # select only decisions by table
                if len(table_df.index) > 0:
                    table_df.to_sql(table, engine, if_exists="append", index=False)
            entries = self.entries_template

    def process_file(self, entries, file):
        type_key = 'fileDesc::sourceDesc::biblStruct::type'
        title_key = 'fileDesc::titleStmt::title'
        date_key = 'fileDesc::sourceDesc::biblStruct::analytic'
        # the first one of these should be the Aktenzeichen
        file_number_key = 'fileDesc::sourceDesc::biblStruct::analytic::idno'
        corpora = TeiReader().read_file(file)  # or read_string
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
            Column('counter_lemma', JSON),
            Column('counter_pos', JSON),
            Column('counter_tag', JSON),
        )
        meta.create_all(engine)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    jureko_processor = JurekoProcessor(config)
    jureko_processor.process()
