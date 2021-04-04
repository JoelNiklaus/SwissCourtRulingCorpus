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
import dask.dataframe as dd
import numpy as np
from root import ROOT_DIR
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.log_utils import get_logger

logger = get_logger(__name__)


class KaggleDatasetCreator(DatasetConstructorComponent):
    """
    Creates the files for a kaggle dataset.
    """

    def __init__(self, config: dict):
        super().__init__(config)

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
        sample_submission = solution.rename(columns={"Expected": "Predicted"})  # rename according to kaggle conventions
        sample_submission['Predicted'] = np.random.choice(selection, size=len(sample_submission))  # set to random value

        # save to csv
        train.to_csv(self.kaggle_csv_subdir / 'train.csv', index_label='Id')
        val.to_csv(self.kaggle_csv_subdir / 'val.csv', index_label='Id')
        test.to_csv(self.kaggle_csv_subdir / 'test.csv', index_label='Id')
        solution.to_csv(self.kaggle_csv_subdir / 'solution.csv', index_label='Id')
        sample_submission.to_csv(self.kaggle_csv_subdir / 'sampleSubmission.csv', index_label='Id')
        logger.info(f"Saved files necessary for competition to {self.kaggle_csv_subdir}")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    kaggle_dataset_creator = KaggleDatasetCreator(config)
    kaggle_dataset_creator.create_dataset()
