"""
Take cleaned German aggregation
keep columns chamber and text
sample 100 decisions from each of the 10 most frequent chambers
shuffle
split into train, validation and test set
remove labels
possible labels: chamber, (date)

"""
import configparser
import pandas as pd
import dask.dataframe as dd
import numpy as np
from root import ROOT_DIR
from scrc.utils.log_utils import get_logger

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
        logger.info("Reading csv file")
        ddf = dd.read_csv(self.clean_csv_subdir / '_de.csv', usecols=['chamber', 'text'])

        seed = 42
        n_most_frequent_chambers = 10
        num_decisions_per_chamber = 100

        value_counts = ddf.chamber.value_counts().compute()  # count number of decisions per chamber
        selection = list(value_counts[:n_most_frequent_chambers].index)  # select n most frequent chambers
        logger.info(f"Computed list of most frequent chambers: {selection}")

        # select only decisions which are of a chamber in the selection
        df = ddf[ddf.chamber.str.contains("|".join(selection), regex=True, na=False)].compute()

        # sample num_decisions_per_chamber from each of the selected chambers
        df = df.sample(n=n_most_frequent_chambers * num_decisions_per_chamber, random_state=seed)
        logger.info(f"Sampled {num_decisions_per_chamber} "
                    f"from each of the {n_most_frequent_chambers} most frequent chambers from the dataset")

        # reset index so that we get nice Ids
        # df = df.reset_index(drop=True)
        df.index = np.arange(len(df))

        # split into train, val and test sets
        train, val, solution = dd.from_pandas(df, npartitions=1).random_split([0.7, 0.1, 0.2], random_state=seed)
        logger.info("Split data into train, val and test set")

        # get pandas dfs again
        train = train.compute()
        val = val.compute()
        solution = solution.compute()

        # create test file
        test = solution.drop('chamber', axis='columns')  # drop label

        # create solution file
        solution = solution.drop('text', axis='columns')  # drop text
        solution = solution.rename(columns={"chamber": "Expected"})  # rename according to kaggle conventions

        # create sampleSubmission file
        sampleSubmission = solution.rename(columns={"Expected": "Predicted"})  # rename according to kaggle conventions
        sampleSubmission['Predicted'] = np.random.choice(selection, size=len(sampleSubmission))  # set to random value

        # save to csv
        train.to_csv(self.kaggle_csv_subdir / 'train.csv', index_label='Id')
        val.to_csv(self.kaggle_csv_subdir / 'val.csv', index_label='Id')
        test.to_csv(self.kaggle_csv_subdir / 'test.csv', index_label='Id')
        solution.to_csv(self.kaggle_csv_subdir / 'solution.csv', index_label='Id')
        sampleSubmission.to_csv(self.kaggle_csv_subdir / 'sampleSubmission.csv', index_label='Id')
        logger.info(f"Saved files necessary for competition to {self.kaggle_csv_subdir}")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    kaggle_dataset_creator = KaggleDatasetCreator(config)
    kaggle_dataset_creator.create_dataset()
