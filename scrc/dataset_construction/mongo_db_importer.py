from pymongo import MongoClient
import subprocess

from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent

import configparser

from root import ROOT_DIR
from scrc.utils.log_utils import get_logger


class MongoDBImporter(DatasetConstructorComponent):
    """
    Imports the csv data to a MongoDB
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.ip = config['mongodb']['ip']
        self.port = config['mongodb']['port']
        self.database = config['mongodb']['database']
        self.collection = config['mongodb']['collection']

    def connect_to_db(self):
        client = MongoClient(f"mongodb://{self.ip}:{self.port}")
        database = client[self.database]
        collection = database[self.collection]

    def import_data(self):
        file_to_import = self.clean_subdir / "_all.csv"

        self.import_file(file_to_import)

    def import_file(self, file_to_import):
        bash_command = f"mongoimport --host {self.ip} --port {self.port} -d {self.database} -c {self.collection} --type csv --file {file_to_import} --headerline"
        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        self.logger.info(output)
        self.logger.error(error)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    kaggle_dataset_creator = MongoDBImporter(config)
    kaggle_dataset_creator.import_data()
