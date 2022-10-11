import gc
from typing import Tuple

import pandas as pd
import datasets
from datasets import concatenate_datasets, Dataset

from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.enums.section import Section
from scrc.utils.log_utils import get_logger

from scrc.utils.main_utils import get_config, print_memory_usage
import scrc.utils.monkey_patch  # prevent memory leak with pandas


class PretrainingDatasetCreator(DatasetCreator):
    """
    Creates a dataset with all the full_text
    """

    def __init__(self, config: dict, debug: bool = True):
        super().__init__(config, debug)
        self.logger = get_logger(__name__)

        self.overwrite_cache = True
        self.split_type = "all_train"
        self.dataset_name = "swiss_caselaw"
        self.feature_cols = [Section.FULL_TEXT]

    def prepare_dataset(self, save_reports, court_string):
        engine = self.get_engine(self.db_scrc)
        data_to_load = {
            "section": True, "file": False, "file_number": False,
            "judgment": False, "citation": False, "lower_court": False
        }
        # we don't use the cache since it is overwritten after each court
        df = self.get_df(engine, data_to_load, court_string=court_string, use_cache=False)
        hf_dataset = datasets.Dataset.from_pandas(df)

        return hf_dataset, None


if __name__ == '__main__':
    config = get_config()

    pretraining_dataset_creator = PretrainingDatasetCreator(config, debug=False)
    pretraining_dataset_creator.create_multiple_datasets(concatenate=True, overview=False, sub_datasets=False,
                                                           save_reports=False)
