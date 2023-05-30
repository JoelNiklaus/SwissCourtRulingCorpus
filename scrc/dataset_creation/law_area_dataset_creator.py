from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.enums.section import Section
from scrc.utils.log_utils import get_logger
import numpy as np
import datasets

from scrc.dataset_creation.report_creator import ReportCreator
from scrc.utils.main_utils import get_config
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
        self.filter_cols = [Section.FACTS]

        self.only_sub_areas = False

        self.labels = ['label']
        self.start_years[Split.TRAIN.value] = 1970
        if self.only_sub_areas:
            self.dataset_name = "law_sub_area_prediction"
        self.metadata = ['year', 'chamber', 'court', 'canton', 'region',
                         'law_area', 'law_sub_area']
        self.delete_row_only_if_all_feature_cols_below_cutoff = False


    def prepare_dataset(self, save_reports, court_string):
        data_to_load = {
            "section": True, "file": True, "file_number": True,
            "judgment": False, "citation": False, "lower_court": False,
            "law_area": True, "law_sub_area": True
        }
        df = self.get_df(self.get_engine(self.db_scrc), data_to_load, court_string=court_string, use_cache=False)
        if df.empty:
            self.logger.warning("No data found")
            return datasets.Dataset.from_pandas(df), []

        df = df.dropna(subset=['law_area'])  # drop empty labels introduced by cleaning before

        if self.only_sub_areas:
            # replace "nan" with np.nan
            df['law_sub_area'] = df['law_sub_area'].replace("nan", np.nan)
            df = df.dropna(subset=['law_sub_area'])
            df = df.rename(columns={'law_sub_area': 'label'})
        else:
            df = df.rename(columns={"law_area": "label"})  # normalize column names

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
    judgment_dataset_creator.create_multiple_datasets(concatenate=True, overview=True, save_reports=True)