from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.utils.log_utils import get_logger

from scrc.utils.main_utils import get_config


class PretrainingDatasetCreator(DatasetCreator):
    """
    Creates a dataset with all the full_text
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.debug = True
        self.split_type = "all_train"
        self.dataset_name = "fscs"
        self.feature_cols = ['full_text']

    def get_dataset(self, feature_col, save_reports):
        df = self.get_df(self.get_engine(self.db_scrc), feature_col)

        return df, None


if __name__ == '__main__':
    config = get_config()

    pretraining_dataset_creator = PretrainingDatasetCreator(config)
    pretraining_dataset_creator.create_dataset(sub_datasets=False, kaggle=False, huggingface=True, save_reports=False)
