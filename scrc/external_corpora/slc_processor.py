import configparser
import glob
from pathlib import Path
from pprint import pprint

import bs4
import spacy
from tqdm import tqdm

from sqlalchemy import MetaData, Table, Column, Integer, String, Date, JSON

import pandas as pd

from root import ROOT_DIR
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.log_utils import get_logger


class SlcProcessor(DatasetConstructorComponent):
    """
    Swiss Legislation Corpus: https://pub.cl.uzh.ch/wiki/public/pacoco/swiss_legislation_corpus
    Obtained by email enquiry
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

    def process(self):
        engine = self.get_engine(self.db_slc)
        self.extract_to_db(engine)

        disable_pipes = ['senter', 'ner', 'attribute_ruler', 'textcat']
        nlp = spacy.load('de_core_news_lg', disable=disable_pipes)
        nlp.max_length = 3000000

        self.logger.info("Running spacy pipeline")
        processed_file_path = self.slc_subdir / f"slc_spacied.txt"
        parts, message = self.compute_remaining_parts(processed_file_path, ["slc"])
        self.logger.info(message)
        for part in parts:
            part_dir = self.create_dir(self.slc_spacy_subdir, part)
            self.run_spacy_pipe(engine, part, part_dir, "", nlp, self.logger)
            self.mark_as_processed(processed_file_path, part)

        self.logger.info("Computing counters")
        processed_file_path = self.slc_subdir / f"slc_counted.txt"
        parts, message = self.compute_remaining_parts(processed_file_path, ["slc"])
        self.logger.info(message)
        for part in parts:
            part_dir = self.slc_spacy_subdir / part
            spacy_vocab = self.load_vocab(part_dir)
            self.compute_counters(engine, part, "", spacy_vocab, part_dir, self.logger)
            self.mark_as_processed(processed_file_path, part)

        self.compute_total_aggregate(engine, ["slc"], "slc", self.slc_subdir, self.logger)

    def extract_to_db(self, engine):
        file_names = [file_path for file_path in glob.glob(str(self.slc_subdir / "30_XML_POSTagged/DE/*"))]
        processed_file_path = self.slc_subdir / f"files_extracted.txt"
        files, message = self.compute_remaining_parts(processed_file_path, file_names)
        self.logger.info(message)

        entries = {'sr': [], 'title': [], 'lang': [], 'text': []}
        i = 1
        for file in tqdm(files, total=len(files)):
            self.process_file(entries, file)
            self.mark_as_processed(processed_file_path, file)
            self.save_to_db(engine, entries, i)
            i += 1
        self.save_to_db(engine, entries, 0)  # save the last chunk (smaller than chunksize)

    def save_to_db(self, engine, entries, i):
        self.create_table(engine)
        if i % self.chunksize == 0:  # save to db in chunks so we don't overload RAM
            df = pd.DataFrame.from_dict(entries)
            df.to_sql("slc", engine, if_exists="append", index=False)
            entries = {'sr': [], 'title': [], 'lang': [], 'text': []}

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
