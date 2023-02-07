from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.enums.section import Section
from scrc.utils.log_utils import get_logger
import numpy as np
import datasets
from typing import Dict

from scrc.dataset_creation.report_creator import ReportCreator
from scrc.utils.main_utils import get_config
from scrc.enums.split import Split

class CourtViewDatasetCreator(DatasetCreator):
    """
    Creates a dataset with the facts as input and the considerations as labels
    """

    def __init__(self, config: dict, debug: bool = True):
        super().__init__(config, debug)
        self.logger = get_logger(__name__)

        self.split_type = "date-stratified"
        self.dataset_name = "court_view_generation"
        self.feature_cols = [Section.FACTS, Section.CONSIDERATIONS]
        self.only_with_origin = False

        self.labels = []
        self.start_years = {Split.TRAIN.value: 1970, Split.VALIDATION.value: 2016, Split.TEST.value: 2018,
                            Split.SECRET_TEST.value: 2023}
        self.metadata = ['year', 'chamber', 'court', 'canton', 'region', 'law_area']


    def prepare_dataset(self, save_reports, court_string):
        data_to_load = {
            "section": True, "file": True, "file_number": True,
            "judgment": False, "citation": False, "lower_court": True,
            "law_area": True, "law_sub_area": False
        }
        df = self.get_df(self.get_engine(self.db_scrc), data_to_load, court_string=court_string, use_cache=False)

        if self.only_with_origin:
            self.dataset_name = "court_view_generation_L2"

            if 'origin_facts' in df.columns:
                df = self.filter_by_num_tokens(df, 'origin_facts')
                df = self.filter_by_num_tokens(df, 'origin_considerations')
            else:
                self.logger.warning("Only_with_origin is set to True, but origin_facts is not in the dataframe")
                # make df empty
                df = df.iloc[0:0]

        if df.empty:
            self.logger.warning("No data found")

        return datasets.Dataset.from_pandas(df), []

    def plot_custom(self, report_creator: ReportCreator, df, split_folder):
        pass


if __name__ == '__main__':
    config = get_config()

    judgment_dataset_creator = CourtViewDatasetCreator(config, debug=False)
    judgment_dataset_creator.create_multiple_datasets(concatenate=True, overview=True, save_reports=True)