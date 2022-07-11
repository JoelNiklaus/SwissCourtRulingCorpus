from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.utils.log_utils import get_logger
import numpy as np

from scrc.utils.main_utils import get_config
from scrc.utils.sql_select_utils import convert_to_binary_judgments


class JudgmentDatasetCreator(DatasetCreator):
    """
    Creates a dataset with the facts or considerations as input and the judgments as labels
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.debug = True
        self.split_type = "date-stratified"
        self.dataset_name = "judgment_prediction"
        self.feature_cols = ['facts', 'considerations']

        self.with_partials = False
        self.with_write_off = False
        self.with_unification = False
        self.with_inadmissible = False
        self.make_single_label = True

    def get_dataset(self, feature_col, save_reports):
        df = self.get_df(self.get_engine(self.db_scrc), feature_col, 'judgments', save_reports)

        # Delete cases with "Nach Einsicht" from the dataset because they are mostly inadmissible or otherwise dismissal
        # => too easily learnable for the model (because of spurious correlation)
        if self.with_inadmissible:
            df = df[~df[feature_col].str.startswith('Nach Einsicht')]

        df = df.dropna(subset=['judgments'])
        df = convert_to_binary_judgments(df, self.with_partials, self.with_write_off, self.with_unification,
                                         self.with_inadmissible, self.make_single_label)
        df = df.dropna(subset=['judgments'])  # drop empty labels introduced by cleaning before

        df = df.rename(columns={"judgments": "label"})  # normalize column names
        labels, _ = list(np.unique(np.hstack(df.label), return_index=True))
        return df, labels


if __name__ == '__main__':
    config = get_config()

    judgment_dataset_creator = JudgmentDatasetCreator(config)
    judgment_dataset_creator.create_dataset(sub_datasets=False, kaggle=False, huggingface=True, save_reports=False)
