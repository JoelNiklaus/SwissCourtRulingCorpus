import glob
from pathlib import Path

import spacy
from tqdm import tqdm

import pandas as pd

from scrc.preprocessing.abstract_preprocessor import AbstractPreprocessor
from scrc.utils.log_utils import get_logger


class ExternalCorpusProcessor(AbstractPreprocessor):
    """
    Swiss Legislation Corpus: https://pub.cl.uzh.ch/wiki/public/pacoco/swiss_legislation_corpus
    Obtained by email enquiry
    """

    # Subclasses should set these
    subdir = None
    spacy_subdir = None
    db = None
    tables_name = None
    tables = None
    entries_template = None
    glob_str = None

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

    def process(self):
        assert self.db  # make sure it has been set
        engine = self.get_engine(self.db)
        self.extract_to_db(engine)

        disable_pipes = ['senter', 'ner', 'attribute_ruler', 'textcat']
        nlp = spacy.load('de_core_news_lg', disable=disable_pipes)
        nlp.max_length = 3000000

        self.logger.info("Running spacy pipeline")
        assert isinstance(self.subdir, Path)
        assert isinstance(self.spacy_subdir, Path)
        processed_file_path = self.subdir / f"{self.tables_name}_spacied.txt"
        parts, message = self.compute_remaining_parts(processed_file_path, self.tables)
        self.logger.info(message)
        for part in parts:
            part_dir = self.create_dir(self.spacy_subdir, part)
            self.run_nlp_pipe(engine, part, part_dir, "", nlp, None, self.logger)
            self.mark_as_processed(processed_file_path, part)

        self.logger.info("Computing counters")
        processed_file_path = self.subdir / f"parts_counted.txt"
        parts, message = self.compute_remaining_parts(processed_file_path, self.tables)
        self.logger.info(message)
        for part in parts:
            part_dir = self.spacy_subdir / part
            spacy_vocab = self.load_vocab(part_dir)
            self.compute_counters(engine, part, "", spacy_vocab, part_dir, self.logger)
            self.mark_as_processed(processed_file_path, part)

        self.compute_total_aggregate(engine, self.tables, self.tables_name, self.subdir, self.logger)

    def extract_to_db(self, engine):
        file_names = [file_path for file_path in glob.glob(str(self.subdir / self.glob_str))]
        processed_file_path = self.subdir / f"files_extracted.txt"
        files, message = self.compute_remaining_parts(processed_file_path, file_names)
        self.logger.info(message)

        entries = self.entries_template
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
            df.to_sql(self.tables_name, engine, if_exists="append", index=False)
            entries = self.entries_template

    def process_file(self, entries, file):
        """
        Subclasses should override this method
        :param entries:
        :param file:
        :return:
        """
        pass

    def create_table(self, engine):
        """
        Subclasses should override this method
        :param engine:
        :return:
        """
        pass
