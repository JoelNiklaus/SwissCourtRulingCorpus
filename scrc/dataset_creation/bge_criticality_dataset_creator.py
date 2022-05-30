from scrc.dataset_creation.criticality_dataset_creator import CriticalityDatasetCreator
from scrc.utils.log_utils import get_logger
import pandas as pd

from scrc.utils.main_utils import get_config


class BgeCriticalityDatasetCreator(CriticalityDatasetCreator):
    """
    Defines a criticality label for each found bger case, based if the case was published as
    bge or not.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # self.dataset_name = "criticality_prediction"
        # self.feature_cols = ['text']  # ['facts', 'considerations', 'text']

    # set criticality labels
    def get_labeled_data(self, bger_df, bge_df):
        self.logger.info(f"Processing labeling of bge_criticality")

        # Include all bger rulings whose file_number can be found in the header of a bge
        # It's not enough no compare date and chamber, there are multiple matching cases
        # There exist around 12'000 rulings with date = 1.1.2020
        # error sources:
        # 1. Regex cannot find correct file number in header
        # 2. languages are different -> different datasets

        # TODO create method comparing bger file numbers to found regex expression in bge
        """
        file_number_match = bger_df.file_number.astype(str).isin(list(bge_df.bge_reference.astype(str)))
        file_number_match_df = bger_df[file_number_match]       
        critical_df = bger_df[file_number_match]
        critical_df['label'] = 'critical'
        non_critical_df = bger_df[~file_number_match]
        non_critical_df['label'] = 'non-critical'
        self.logger.info(f"# critical decisions: {len(critical_df.index)}")
        self.logger.info(f"# non-critical decisions: {len(non_critical_df.index)}")
        return critical_df.append(non_critical_df)
        """
        return bger_df


if __name__ == '__main__':
    config = get_config()

    bge_criticality_dataset_creator = BgeCriticalityDatasetCreator(config)
    bge_criticality_dataset_creator.create_dataset()
