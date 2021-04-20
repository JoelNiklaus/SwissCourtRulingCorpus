import gc
import multiprocessing
import sys
from pathlib import Path
import glob
from time import sleep
from memory_profiler import profile
import os, psutil

import spacy
from spacy.vocab import Vocab
from tqdm import tqdm
import pandas as pd
import configparser
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from root import ROOT_DIR
from scrc.utils.decorators import slack_alert
from scrc.utils.log_utils import get_logger

# import scrc.utils.monkey_patch  # prevent memory leak with pandas

# IMPORTANT: make sure you download these models first with: python -m spacy download de_dep_news_trf
import de_core_news_lg, fr_core_news_lg, it_core_news_lg

# TODO use postgres to store all this stuff:
#  https://pythonspeed.com/articles/indexing-pandas-sqlite/
#  https://pythonspeed.com/articles/pandas-sql-chunking/
from scrc.utils.main_utils import get_file_gen
from scrc.utils.slack_util import post_message_to_slack

# TODO find out how to deal with deadlocks
#  A: prevent deadlock
#  B: restart program when deadlock is detected
#  C: for diagnosis print threaddump

class SpacyPipelineRunner(DatasetConstructorComponent):
    """
    Runs the entire spacy pipeline for each text and saves it into the MongoDB.
    This brings the advantage, that we have the heavy computation done in advance,
    and can then use the spacy objects directly in our analysis.

    Here is a very good resource to reduce memory consumption: https://pythonspeed.com/memory/
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.models = {
            'de': 'de_core_news_lg',
            'fr': 'fr_core_news_lg',
            'it': 'it_core_news_lg'
        }
        # tag, pos and lemma are enough for now
        self.disable_pipes = ['senter', 'ner', 'attribute_ruler', 'textcat']
        self.active_model = None


    @staticmethod
    def load_spacy_model(model_name, disable_pipes):
        return spacy.load(model_name, disable=disable_pipes)

    @slack_alert
    def run_pipeline(self):
        self.logger.info("Started running spacy pipeline on the texts")
        for lang in self.languages:
            self.logger.info(f"Processing language {lang}")

            in_lang_dir = self.split_subdir / lang  # input dir
            out_lang_dir = self.create_dir(self.spacy_subdir, lang)  # output dir

            chamber_list = [Path(chamber).stem for chamber in glob.glob(f"{str(in_lang_dir)}/*.parquet")]
            self.logger.info(f"Found {len(chamber_list)} chambers in total")

            chambers_processed_path = out_lang_dir / "chambers_processed.txt"
            if not chambers_processed_path.exists():
                chambers_processed_path.touch()
            chambers_processed = chambers_processed_path.read_text().split("\n")
            self.logger.info(f"Found {len(chambers_processed)} chamber(s) already processed: {chambers_processed}")

            chambers_not_yet_processed = set(chamber_list) - set(chambers_processed)
            self.logger.info(
                f"Still {len(chambers_not_yet_processed)} chamber(s) remaining to process: {chambers_not_yet_processed}")

            if chambers_not_yet_processed: # if we still actually have some chambers remaining
                self.load_language_model(lang, out_lang_dir)

            self.process_chambers(chambers_not_yet_processed, chambers_processed_path, in_lang_dir, out_lang_dir)

        self.logger.info("Finished running spacy pipeline on the texts")

    def load_language_model(self, lang, out_lang_dir):
        self.logger.info("Loading spacy model")
        self.active_model = self.load_spacy_model(self.models[lang], self.disable_pipes)
        # increase max length for long texts: Can lead to memory allocation errors for parser and ner
        self.active_model.max_length = 3000000
        vocab_path = out_lang_dir / f"_vocab.spacy"
        if vocab_path.exists():
            vocab = Vocab().from_disk(str(vocab_path))
            self.active_model.vocab = vocab

    def process_chambers(self, chambers_not_yet_processed, chambers_processed_path, in_lang_dir, out_lang_dir):
        for chamber in chambers_not_yet_processed:
            self.logger.info(f"Processing chamber {chamber}")

            # according to docs you should aim for a partition size of 100MB
            # 1 court decision takes approximately between around 10KB and 100KB of RAM when loaded into memory
            # The spacy doc takes about 25x the size of a court decision
            self.run_spacy_pipeline(in_lang_dir, out_lang_dir, chamber)

            with chambers_processed_path.open("a") as f:
                f.write(chamber + "\n")

    @profile
    def run_spacy_pipeline(self, in_lang_dir, out_lang_dir, chamber):
        """
        Creates and saves the docs generated by the spacy pipeline.
        """
        file_gen = get_file_gen(in_lang_dir / (chamber + ".parquet"))
        if file_gen:
            for chunk, df in file_gen:
                chamber_dir = self.create_dir(out_lang_dir, chamber)
                path = chamber_dir / f"part.{chunk}.parquet"
                if not path.exists():  # make sure we did not compute it before already
                    self.logger.info(f"Processing chunk {chunk} and saving it to {path}")
                    texts = df.text.tolist()
                    # batch_size = max(int(len(texts) / self.num_cpus), 1) # a high batch_size can lead to lots of allocated memory
                    docs = tqdm(self.active_model.pipe(texts, n_process=-1, batch_size=1), total=len(texts))
                    df['spacy_doc_bytes'] = [doc.to_bytes() for doc in docs]
                    df['num_tokens'] = [len(doc) for doc in docs]
                    df.to_parquet(path, index=False)

                    del df, docs

                    self.active_model.vocab.to_disk(out_lang_dir / f"_vocab.spacy")

                    gc.collect()
                    sleep(2)  # sleep(2) is required to allow measurement of the garbage collector

            memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
            message = f"Your running process is currently using {memory_usage:.3f} GB of memory"
            print(message)
            try:
                post_message_to_slack(message)
            except:
                print("Could not send message to slack: ", sys.exc_info())





if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    spacy_pipeline_runner = SpacyPipelineRunner(config)
    spacy_pipeline_runner.run_pipeline()
