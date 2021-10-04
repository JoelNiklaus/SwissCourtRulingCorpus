import configparser
from pathlib import Path

import bs4

from sqlalchemy import MetaData, Table, Column, Integer, String, JSON

from root import ROOT_DIR
from scrc.preprocessors.external_corpora.external_corpus_processor import ExternalCorpusProcessor
from scrc.utils.log_utils import get_logger


class SlcProcessor(ExternalCorpusProcessor):
    """
    Swiss Legislation Corpus: https://pub.cl.uzh.ch/wiki/public/pacoco/swiss_legislation_corpus
    Obtained by email enquiry
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.subdir = self.slc_subdir
        self.spacy_subdir = self.slc_spacy_subdir
        self.db = self.db_slc
        self.tables_name = "slc"
        self.tables = ["slc"]
        self.entries_template = {'sr': [], 'title': [], 'lang': [], 'text': []}
        self.glob_str = "30_XML_POSTagged/DE/*.xml"

    def process_file(self, entries, file):
        soup = bs4.BeautifulSoup(Path(file).read_text(), "lxml-xml")

        if not soup:
            return
        text = soup.find('text')
        if not text:
            return
        meta = text.find('meta')
        entries['sr'].append(meta.find('sr').text)
        entries['title'].append(meta.find('name').text)
        entries['lang'].append(meta.find('lang').text)

        tokens = []
        paragraphs = text.find('doc').find_all('p')
        if not paragraphs:
            return
        for p in paragraphs:
            words = p.find_all('t')
            for word in words:
                tokens.append(word['word'])

        entries['text'].append(tokens)

    def create_table(self, engine):
        meta = MetaData()
        table = Table(
            "slc", meta,
            Column('id', Integer, primary_key=True),
            Column('sr', String),
            Column('title', String),
            Column('lang', String),
            Column('text', String),
            Column('num_tokens', Integer),
            Column('counter_lemma', JSON),
            Column('counter_pos', JSON),
            Column('counter_tag', JSON),
        )
        meta.create_all(engine)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    slc_processor = SlcProcessor(config)
    slc_processor.process()
