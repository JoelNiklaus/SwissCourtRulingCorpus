from root import ROOT_DIR
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


class DatasetConstructorComponent:
    """
    Extracts the textual and meta information from the court rulings files and saves it in csv files for each court
    and in one for all courts combined
    """

    def __init__(self, config: dict):
        self.data_dir = ROOT_DIR / config['dir']['data_dir']

        self.spiders_dir = self.data_dir / config['dir']['spiders_subdir']
        self.spiders_dir.mkdir(parents=True, exist_ok=True)  # create folder if it does not exist yet

        self.csv_dir = self.data_dir / config['dir']['csv_subdir']
        self.raw_csv_subdir = self.csv_dir / config['dir']['raw_csv_subdir']
        self.raw_csv_subdir.mkdir(parents=True, exist_ok=True)  # create folder if it does not exist yet

        self.clean_csv_subdir = self.csv_dir / config['dir']['clean_csv_subdir']
        self.clean_csv_subdir.mkdir(parents=True, exist_ok=True)  # create folder if it does not exist yet

        self.kaggle_csv_subdir = self.csv_dir / config['dir']['kaggle_csv_subdir']
        self.kaggle_csv_subdir.mkdir(parents=True, exist_ok=True)  # create folder if it does not exist yet
