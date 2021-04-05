from root import ROOT_DIR
import pandas as pd

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
        self.data_dir = ROOT_DIR / config['dir']['data_dir']

        self.spiders_dir = self.data_dir / config['dir']['spiders_subdir']
        self.spiders_dir.mkdir(parents=True, exist_ok=True)  # create folder if it does not exist yet

        self.raw_subdir = self.data_dir / config['dir']['raw_subdir']
        self.raw_subdir.mkdir(parents=True, exist_ok=True)  # create folder if it does not exist yet

        self.clean_subdir = self.data_dir / config['dir']['clean_subdir']
        self.clean_subdir.mkdir(parents=True, exist_ok=True)  # create folder if it does not exist yet

        self.kaggle_subdir = self.data_dir / config['dir']['kaggle_subdir']
        self.kaggle_subdir.mkdir(parents=True, exist_ok=True)  # create folder if it does not exist yet
