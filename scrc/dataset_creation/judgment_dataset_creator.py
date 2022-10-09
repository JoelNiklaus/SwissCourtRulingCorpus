from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.enums.section import Section
from scrc.utils.log_utils import get_logger
import numpy as np
import datasets

from scrc.utils.main_utils import get_config
from scrc.utils.sql_select_utils import convert_to_binary_judgments

import os
from tqdm import tqdm
from root import ROOT_DIR
from scrc.utils.court_names import get_all_courts, get_issue_courts, get_error_courts


class JudgmentDatasetCreator(DatasetCreator):
    """
    Creates a dataset with the facts or considerations as input and the judgments as labels
    """

    def __init__(self, config: dict, debug: bool = True):
        super().__init__(config, debug)
        self.logger = get_logger(__name__)

        self.split_type = "date-stratified"
        self.dataset_name = "judgment_prediction"
        self.feature_cols = [Section.FACTS, Section.CONSIDERATIONS]

        self.with_partials = False
        self.with_write_off = False
        self.with_unification = False
        self.with_inadmissible = False
        self.make_single_label = True
        self.labels = ['label']

    def prepare_dataset(self, save_reports, court_string):
        data_to_load = {
            "section": True, "file": True, "file_number": True,
            "judgment": True, "citation": False, "lower_court": True
        }
        df = self.get_df(self.get_engine(self.db_scrc), data_to_load, court_string=court_string, use_cache=False)
        if df.empty:
            self.logger.warning("No data found")
            return datasets.Dataset.from_pandas(df), []
        df = df.dropna(subset=['judgments'])
        df = convert_to_binary_judgments(df, self.with_partials, self.with_write_off, self.with_unification,
                                         self.with_inadmissible, self.make_single_label)
        df = df.dropna(subset=['judgments'])  # drop empty labels introduced by cleaning before
        df = df.rename(columns={"judgments": "label"})  # normalize column names
        labels, _ = list(np.unique(np.hstack(df.label), return_index=True))
        return datasets.Dataset.from_pandas(df), [labels]

    def plot_custom(self, df, split_folder, folder):
        self.plot_labels(df, split_folder, label_name='label')


if __name__ == '__main__':
    config = get_config()

    judgment_dataset_creator = JudgmentDatasetCreator(config, debug=True)
    judgment_dataset_creator.create_multiple_datasets(concatenate=True, overview=True)
