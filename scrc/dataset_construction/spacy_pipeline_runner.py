import gc
from pathlib import Path
import glob

import spacy
from dask.diagnostics import ProgressBar
from tqdm import tqdm
import pandas as pd
import dask.dataframe as dd
import configparser
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from root import ROOT_DIR
from scrc.utils.decorators import slack_alert
from scrc.utils.log_utils import get_logger

# import scrc.utils.monkey_patch  # prevent memory leak with pandas

# IMPORTANT: make sure you download these models first with: python -m spacy download de_dep_news_trf
import de_core_news_lg, fr_core_news_lg, it_core_news_lg


class SpacyPipelineRunner(DatasetConstructorComponent):
    """
    Runs the entire spacy pipeline for each text and saves it into the MongoDB.
    This brings the advantage, that we have the heavy computation done in advance,
    and can then use the spacy objects directly in our analysis.
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
        self.disable_pipes = ['tagger', 'parser', 'senter', 'ner', 'attribute_ruler', 'textcat']
        self.active_model = None

    def load_spacy_model(self, model_name, disable_pipes):
        return spacy.load(model_name, disable=disable_pipes)

    @slack_alert
    def run_pipeline(self):
        self.logger.info("Started running spacy pipeline on the texts")
        for lang in self.languages:
            self.logger.info(f"Processing language {lang}")
            self.logger.info("Loading spacy model")
            self.active_model = self.load_spacy_model(self.models[lang], self.disable_pipes)
            self.active_model.max_length = 2000000  # increase max length for long texts

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

            self.process_chamber(chambers_not_yet_processed, chambers_processed_path, in_lang_dir, out_lang_dir)

            self.active_model.vocab.to_disk(out_lang_dir / f"{lang}_vocab.spacy")

        self.logger.info("Finished running spacy pipeline on the texts")

    def process_chamber(self, chambers_not_yet_processed, chambers_processed_path, in_lang_dir, out_lang_dir):
        for chamber in chambers_not_yet_processed:
            df = pd.read_parquet(in_lang_dir / (chamber + ".parquet"))
            self.logger.info(f"Processing the {len(df.index)} decisions from chamber {chamber}")

            # according to docs you should aim for a partition size of 100MB
            # 1 court decision takes approximately between around 10KB and 100KB of RAM when loaded into memory
            # The spacy doc takes about 25x the size of a court decision
            ddf = dd.from_pandas(df, chunksize=2000)
            ddf['spacy_doc_bytes'] = 0.0
            ddf = ddf.map_partitions(self.run_spacy_pipeline, meta=ddf)  # compute for every partition
            # ddf = ddf.drop(['html_raw', 'pdf_raw', 'text'], axis='columns')  # remove old cols
            ddf.to_parquet(out_lang_dir / (chamber + ".parquet"), write_index=False,
                           engine='pyarrow')  # save to filesystem

            ddf.compute(scheduler='threads')

            with chambers_processed_path.open("a") as f:
                f.write(chamber + "\n")

            gc.collect()

    def run_spacy_pipeline(self, df):
        run_spacy_pipe = self.active_model.pipe(list(df['text']), n_process=-1, batch_size=1)
        df['spacy_doc_bytes'] = [doc.to_bytes() for doc in tqdm(run_spacy_pipe, total=len(df.index))]
        print(df.memory_usage(deep=True))
        return df


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    spacy_pipeline_runner = SpacyPipelineRunner(config)
    spacy_pipeline_runner.run_pipeline()
