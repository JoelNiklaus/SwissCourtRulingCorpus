import configparser
import glob
from pathlib import Path

import bs4
import requests
import spacy
from tqdm import tqdm

from tei_reader import TeiReader

from sqlalchemy import MetaData, Table, Column, Integer, String, Date, JSON

import pandas as pd
import xmltodict

from root import ROOT_DIR
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.log_utils import get_logger


class WikipediaProcessor(DatasetConstructorComponent):
    """
    Inspired by: https://github.com/t-systems-on-site-services-gmbh/german-wikipedia-text-corpus
    Used https://github.com/attardi/wikiextractor

    Downloaded from this url with curl:
    https://dumps.wikimedia.org/dewiki/20210520/dewiki-20210520-pages-articles-multistream.xml.bz2

    Extracted with WikiExtractor:
    python -m wikiextractor.WikiExtractor --no-templates wikipedia.xml.bz2
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

    def process(self):
        engine = self.get_engine(self.db_wikipedia)
        self.extract_to_db(engine)

        disable_pipes = ['senter', 'ner', 'attribute_ruler', 'textcat']
        nlp = spacy.load('de_core_news_lg', disable=disable_pipes)
        nlp.max_length = 3000000

        self.logger.info("Running spacy pipeline")
        processed_file_path = self.wikipedia_subdir / f"wiki_spacied.txt"
        parts, message = self.compute_remaining_parts(processed_file_path, ["wikipedia"])
        self.logger.info(message)
        for part in parts:
            part_dir = self.create_dir(self.wikipedia_spacy_subdir, part)
            self.run_spacy_pipe(engine, part, part_dir, "", nlp, self.logger)
            self.mark_as_processed(processed_file_path, part)

        self.logger.info("Computing counters")
        processed_file_path = self.wikipedia_subdir / f"wiki_counted.txt"
        parts, message = self.compute_remaining_parts(processed_file_path, ["wikipedia"])
        self.logger.info(message)
        for part in parts:
            part_dir = self.wikipedia_spacy_subdir / part
            spacy_vocab = self.load_vocab(part_dir)
            self.compute_counters(engine, part, "", spacy_vocab, part_dir, self.logger)
            self.mark_as_processed(processed_file_path, part)

        self.compute_total_aggregate(engine, ["wikipedia"], "wikipedia", self.wikipedia_subdir, self.logger)

    def extract_to_db(self, engine):
        file_names = [file_path for file_path in glob.glob(str(self.wikipedia_subdir / "xml/**/*"))]
        processed_file_path = self.wikipedia_subdir / f"files_extracted.txt"
        files, message = self.compute_remaining_parts(processed_file_path, file_names)
        self.logger.info(message)

        entries = {'wiki_id': [], 'title': [], 'url': [], 'text': []}
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
            df.to_sql("wikipedia", engine, if_exists="append", index=False)
            entries = {'wiki_id': [], 'title': [], 'url': [], 'text': []}

    def process_file(self, entries, file):
        text = Path(file).read_text()
        soup = bs4.BeautifulSoup(text, "lxml-xml")
        docs = soup.find_all('doc')
        for doc in docs:
            entries['wiki_id'].append(int(doc['id']))
            entries['title'].append(doc['title'])
            entries['url'].append(doc['url'])
            entries['text'].append(doc.text)

    def create_table(self, engine):
        meta = MetaData()
        table = Table(
            "wikipedia", meta,
            Column('id', Integer, primary_key=True),
            Column('wiki_id', Integer),
            Column('title', String),
            Column('url', String),
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

    wikipedia_processor = WikipediaProcessor(config)
    wikipedia_processor.process()
