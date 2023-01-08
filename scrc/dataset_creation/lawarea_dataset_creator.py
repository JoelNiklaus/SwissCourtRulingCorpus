from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.enums.section import Section
from scrc.utils.log_utils import get_logger
import numpy as np
import datasets
import os
from root import ROOT_DIR
import json

from scrc.dataset_creation.report_creator import ReportCreator
from scrc.utils.main_utils import get_config
from scrc.utils.sql_select_utils import convert_to_binary_judgments
from scrc.enums.split import Split


class LawAreaDatasetCreator(DatasetCreator):
    """
    Creates a dataset with the facts or considerations as input and the law areas as labels
    """

    def __init__(self, config: dict, debug: bool = True):
        super().__init__(config, debug)
        self.logger = get_logger(__name__)

        self.split_type = "date-stratified"
        self.dataset_name = "law_area_prediction"
        self.feature_cols = [Section.FACTS, Section.CONSIDERATIONS]

        self.with_partials = False
        self.with_write_off = False
        self.with_unification = False
        self.with_inadmissible = False
        self.make_single_label = True
        self.labels = ['label']
        self.start_years = {Split.TRAIN.value: 1970, Split.VALIDATION.value: 2016, Split.TEST.value: 2018,
                            Split.SECRET_TEST.value: 2023}

    def add_law_area_labels(self, df):
        # load law area labels
        with open(os.path.join(ROOT_DIR, 'legal_info/chamber_to_area.json'), 'r') as f:
            chamber_to_area = json.load(f)
        # add law area labels
        df['law_area'] = df['chamber'].map(chamber_to_area)
        return df

    def prepare_dataset(self, save_reports, court_string):
        data_to_load = {
            "section": True, "file": True, "file_number": True,
            "judgment": False, "citation": False, "lower_court": False
        }
        df = self.get_df(self.get_engine(self.db_scrc), data_to_load, court_string=court_string, use_cache=False)
        if df.empty:
            self.logger.warning("No data found")
            return datasets.Dataset.from_pandas(df), []
        df = self.add_law_area_labels(df)
        df = df.dropna(subset=['law_area'])  # drop empty labels introduced by cleaning before
        df = df.rename(columns={"law_area": "label"})  # normalize column names
        # drop col "legal_area"
        df = df.drop(columns=['legal_area'])
        # make law_area to the 9th column
        cols = df.columns.tolist()
        cols = cols[:8] + cols[-1:] + cols[8:-1]
        df = df[cols]
        try:
            labels, _ = list(np.unique(np.hstack(df.label), return_index=True))
        except ValueError:
            labels = []
        return datasets.Dataset.from_pandas(df), [labels]

    def plot_custom(self, report_creator: ReportCreator, df, split_folder):
        report_creator.plot_label_ordered(df, label_name='label')


if __name__ == '__main__':
    config = get_config()

    judgment_dataset_creator = LawAreaDatasetCreator(config, debug=False)
    judgment_dataset_creator.create_multiple_datasets(["CH_BGer"], concatenate=False, overview=True, save_reports=True)