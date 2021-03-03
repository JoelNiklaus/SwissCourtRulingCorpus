"""
Take cleaned German aggregation
keep columns court_id, file_number, file_number_additional, date, html_clean and pdf_clean
shuffle
split into train, validation and test set
remove labels
possible labels: court_id, (date)

"""
import configparser
import pandas as pd
from root import ROOT_DIR
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import court_keys

logger = get_logger(__name__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


class KaggleDatasetCreator:
    """
    Creates the files for a kaggle dataset.
    """

    def __init__(self, config: dict):
        self.data_dir = ROOT_DIR / config['dir']['data_dir']
        self.courts_dir = self.data_dir / config['dir']['courts_subdir']
        self.csv_dir = self.data_dir / config['dir']['csv_subdir']
        self.raw_csv_subdir = self.csv_dir / config['dir']['raw_csv_subdir']
        self.clean_csv_subdir = self.csv_dir / config['dir']['clean_csv_subdir']
        self.kaggle_csv_subdir = self.csv_dir / config['dir']['kaggle_csv_subdir']
        self.kaggle_csv_subdir.mkdir(parents=True, exist_ok=True)  # create output folder if it does not exist yet

    def create_dataset(self):
        dtype_dict = {key: 'string' for key in court_keys}  # create dtype_dict from court keys
        df = pd.read_csv(self.clean_csv_subdir / '_de.csv', dtype=dtype_dict)
        logger.info("Finished reading csv file")
        # print(df.describe().compute())

        # find duplicates
        # df = df.sort_values(by=['file_number'])
        # duplicates = df[df[['file_number']].duplicated(keep=False)]  # find duplicates
        # print(duplicates[['file_number', 'url']])

        df = df[['court_id', 'html_clean', 'pdf_clean']]  # select columns

        # split into train, val and test sets
        seed = 42
        train, val, test = df.random_split([0.7, 0.1, 0.2], random_state=seed)

        test = test.drop('court_id', axis='columns')  # drop label

        # shuffle sets
        train = train.sample(frac=1, random_state=seed)
        val = val.sample(frac=1, random_state=seed)
        test = test.sample(frac=1, random_state=seed)

        # save to csv
        train.to_csv(self.kaggle_csv_subdir / 'train.csv', index_label='id')
        val.to_csv(self.kaggle_csv_subdir / 'val.csv', index_label='id')
        test.to_csv(self.kaggle_csv_subdir / 'test.csv', index_label='id')


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    kaggle_dataset_creator = KaggleDatasetCreator(config)
    kaggle_dataset_creator.create_dataset()
