import json

from root import ROOT_DIR
import pandas as pd

from pymongo import MongoClient

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)  # set to None to impose no limit


class DatasetConstructorComponent:
    """
    Extracts the textual and meta information from the court rulings files and saves it in csv files for each court
    and in one for all courts combined
    """

    def __init__(self, config: dict):
        self.languages = json.loads(config['general']['languages'])
        self.chunksize = int(config['general']['chunksize'])

        self.data_dir = self.create_dir(ROOT_DIR, config['dir']['data_dir'])
        self.spiders_dir = self.create_dir(self.data_dir, config['dir']['spiders_subdir'])
        self.raw_subdir = self.create_dir(self.data_dir, config['dir']['raw_subdir'])
        self.clean_subdir = self.create_dir(self.data_dir, config['dir']['clean_subdir'])
        self.split_subdir = self.create_dir(self.data_dir, config['dir']['split_subdir'])
        self.spacy_subdir = self.create_dir(self.data_dir, config['dir']['spacy_subdir'])
        self.kaggle_subdir = self.create_dir(self.data_dir, config['dir']['kaggle_subdir'])

        self.ip = config['mongodb']['ip']
        self.port = config['mongodb']['port']
        self.database = config['mongodb']['database']

    def create_dir(self, parent_dir, dir_name):
        dir = parent_dir / dir_name
        dir.mkdir(parents=True, exist_ok=True)  # create folder if it does not exist yet
        return dir

    def get_db(self):
        return MongoClient(self.ip, int(self.port))[self.database]
