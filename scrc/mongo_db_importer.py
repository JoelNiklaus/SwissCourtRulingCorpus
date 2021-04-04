from pymongo import MongoClient
import subprocess

IP = "172.17.0.2"
PORT = 27017
DB = "scrc"
COLLECTION = "rulings"

client = MongoClient(f"mongodb://{IP}:{PORT}")
database = client.scrc
collection = database.rulings

import configparser

from root import ROOT_DIR
from scrc.utils.log_utils import get_logger

logger = get_logger(__name__)


class MongoDBImporter:
    """
    Imports the csv data to a MongoDB
    """

    def __init__(self, config: dict):
        self.data_dir = ROOT_DIR / config['dir']['data_dir']
        self.courts_dir = self.data_dir / config['dir']['courts_subdir']
        self.csv_dir = self.data_dir / config['dir']['csv_subdir']
        self.raw_csv_subdir = self.csv_dir / config['dir']['raw_csv_subdir']
        self.clean_csv_subdir = self.csv_dir / config['dir']['clean_csv_subdir']

    def import_data(self):
        file_to_import = self.clean_csv_subdir / "_all.csv"

        bash_command = f"mongoimport --host={IP} -d {DB} -c {COLLECTION} --type csv --file {file_to_import} --headerline"
        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        logger.info(output)
        logger.error(error)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    kaggle_dataset_creator = MongoDBImporter(config)
    kaggle_dataset_creator.import_data()
