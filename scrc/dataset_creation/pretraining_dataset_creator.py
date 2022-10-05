import gc

import pandas as pd
import datasets
from datasets import concatenate_datasets

from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.enums.section import Section
from scrc.utils.log_utils import get_logger

from scrc.utils.main_utils import get_config, print_memory_usage
import scrc.utils.monkey_patch  # prevent memory leak with pandas


class PretrainingDatasetCreator(DatasetCreator):
    """
    Creates a dataset with all the full_text
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.debug = True
        self.overwrite_cache = True
        self.split_type = "all_train"
        self.dataset_name = "swiss_caselaw"
        self.feature_cols = [Section.FULL_TEXT]

    def prepare_dataset(self, save_reports) -> datasets.Dataset:
        engine = self.get_engine(self.db_scrc)
        court_strings = next(self.select(engine, "court", "court_string", None))["court_string"].tolist()
        hf_dataset = datasets.Dataset.from_pandas(pd.DataFrame())  # init empty dataset

        data_to_load = {
            "section": True, "file": False, "file_number": False,
            "judgment": False, "citation": False, "lower_court": False
        }

        for court_string in court_strings:
            # we don't use the cache since it is overwritten after each court
            df = self.get_df(engine, data_to_load, court_string=court_string, use_cache=False)

            print_memory_usage([df, hf_dataset])

            self.logger.info("Concatenating datasets")
            hf_dataset = concatenate_datasets([hf_dataset, datasets.Dataset.from_pandas(df)])

            del df
            gc.collect()

        return hf_dataset, None


if __name__ == '__main__':
    config = get_config()

    pretraining_dataset_creator = PretrainingDatasetCreator(config)
    pretraining_dataset_creator.create_dataset(sub_datasets=False, kaggle=False, save_reports=False)
