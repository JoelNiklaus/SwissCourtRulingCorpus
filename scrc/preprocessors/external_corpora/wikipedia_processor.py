import configparser
from pathlib import Path

import bs4
from sqlalchemy import MetaData, Table, Column, Integer, String, JSON

from root import ROOT_DIR
from scrc.preprocessors.external_corpora.external_corpus_processor import ExternalCorpusProcessor
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config


class WikipediaProcessor(ExternalCorpusProcessor):
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

        self.subdir = self.wikipedia_subdir
        self.spacy_subdir = self.wikipedia_spacy_subdir
        self.db = self.db_wikipedia
        self.tables_name = "wikipedia"
        self.tables = ["wikipedia"]
        self.entries_template = {'wiki_id': [], 'title': [], 'url': [], 'text': []}
        self.glob_str = "xml/**/*"

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
    config = get_config()

    wikipedia_processor = WikipediaProcessor(config)
    wikipedia_processor.process()
