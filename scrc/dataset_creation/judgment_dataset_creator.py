from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.enums.section import Section
from scrc.utils.log_utils import get_logger
import numpy as np
import datasets

from scrc.dataset_creation.report_creator import ReportCreator
from scrc.utils.main_utils import get_config
from scrc.utils.sql_select_utils import convert_to_binary_judgments
from scrc.enums.split import Split


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
        self.start_years = {Split.TRAIN.value: 1970, Split.VALIDATION.value: 2016, Split.TEST.value: 2018,
                            Split.SECRET_TEST.value: 2023}
        self.metadata = ['year', 'chamber', 'court', 'canton', 'region',
                         'law_area', 'law_sub_area']

    def prepare_dataset(self, save_reports, court_string):
        data_to_load = {
            "section": True, "file": True, "file_number": True,
            "judgment": True, "citation": False, "lower_court": True,
            "law_area": True, "law_sub_area": True
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
        try:
            labels, _ = list(np.unique(np.hstack(df.label), return_index=True))
        except ValueError:
            labels = []
        return datasets.Dataset.from_pandas(df), [labels]

    def plot_custom(self, report_creator: ReportCreator, df, split_folder):
        report_creator.plot_label_ordered(df, label_name='label')


if __name__ == '__main__':
    config = get_config()

    judgment_dataset_creator = JudgmentDatasetCreator(config, debug=False)
    judgment_dataset_creator.create_multiple_datasets(concatenate=False, overview=True, save_reports=True)