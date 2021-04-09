import multiprocessing
from pprint import pprint
import gridfs
from spacy.tokens import Doc
from spacy.vocab import Vocab
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent

import configparser

from root import ROOT_DIR
from scrc.utils.log_utils import get_logger

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

        disable_pipes = ['parser', 'textcat']

        self.models = {
            'de': de_core_news_lg.load(disable=disable_pipes),
            # 'fr': fr_core_news_lg.load(disable=disable_pipes),
            'it': it_core_news_lg.load(disable=disable_pipes)
        }
        self.active_lang_model = None

        self.active_collection = None

    def run_pipeline(self):
        num_cpus = multiprocessing.cpu_count()

        db = self.get_db()
        print("Connected to MongoDB")
        for lang in self.languages:
            self.active_lang_model = self.models[lang]  # retrieve the spacy model
            self.active_lang_model.max_length = 2000000  # increase max length for long texts

            self.active_collection = db[lang]
            query = {"court": "TI_TCA"}
            cursor = self.active_collection.find(query, {"_id": 1, "text": 1, "spacy_doc": 1}).limit(
                5)  # fetch only texts
            num_docs = self.active_collection.count_documents(query)
            assert num_docs > 0
            generator = self.cursor_to_generator(cursor)

            self.logger.info("Running spacy pipeline on the texts")
            for doc, id in tqdm(
                    self.active_lang_model.pipe(generator, n_process=-1, batch_size=1, as_tuples=True),
                    total=num_docs):
                self.active_collection.update_one({"_id": id}, {"$set": {"spacy_doc": doc.to_bytes()}})

            filename = "it_vocab_bytes"
            self.logger.info(f"Storing the vocab on GridFS under filename {filename}")
            fs = gridfs.GridFS(db)
            fs.put(self.active_lang_model.vocab.to_bytes(), filename=filename)

            # reconstruct vocab
            vocab_bytes = fs.find_one({"filename": filename}).read()
            vocab = Vocab.from_bytes(vocab_bytes)
            # reconstruct doc
            document = self.active_collection.find_one({"court": "TI_TCA"})
            doc = Doc(vocab).from_bytes(document['spacy_doc'])


    def cursor_to_generator(self, cursor):
        self.logger.info("Transforming cursor to generator")
        for item in cursor:
            yield item['text'], item['_id']


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    spacy_pipeline_runner = SpacyPipelineRunner(config)
    spacy_pipeline_runner.run_pipeline()
