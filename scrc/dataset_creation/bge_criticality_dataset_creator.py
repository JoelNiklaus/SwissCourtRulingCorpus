from scrc.dataset_creation.criticality_dataset_creator import CriticalityDatasetCreator
from root import ROOT_DIR
from pathlib import Path
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
        """give each bger ruling a label critical or non-critical depending on whether their
        file number was extracted in a bge"""

        self.logger.info(f"Processing labeling of bge_criticality")
        self.logger.info(f"# there are {len(bger_df.index)} bger decisions")
        self.logger.info(f"# there are {len(bge_df.index)} bge decisions")
        # Include all bger rulings whose file_number can be found in the header of a bge
        # It's not enough no compare date and chamber, there are multiple matching cases
        # error sources:
        # 1. Regex cannot find correct file number in header
        # 2. languages are different -> different datasets
        bge_references_file_path: Path = ROOT_DIR / 'data' / 'progress' / "bge_references_found.txt"
        if not bge_references_file_path.exists():
            bge_references_file_path.touch()
        bge_references = bge_references_file_path.read_text().strip().split("\n")
        file_number_match = bger_df.file_number.astype(str).isin(list(bge_references))
        critical_df = bger_df[file_number_match]
        critical_df['label'] = 'critical'
        non_critical_df = bger_df[~file_number_match]
        non_critical_df['label'] = 'non-critical'
        self.logger.info(f"# critical decisions: {len(critical_df.index)}")
        self.logger.info(f"# non-critical decisions: {len(non_critical_df.index)}")
        self.calculate_label_coverage(bge_references, file_number_match, critical_df, bger_df)
        return critical_df.append(non_critical_df)

    def calculate_label_coverage(self, bge_references, file_number_match, critical_df, bger_df):
        """Calculate some numbers on how many cases could be labeled correctly and hwo many are still missing"""
        self.logger.info(f"there were {len(bge_references)} references extracted")
        bge_references = set(bge_references)
        self.logger.info(f"{len(bge_references)} of the entries were unique")
        # get references which were extracted but not found in bger cases
        extracted_and_found = list(critical_df.file_number.astype(str))
        new_list = [decision for decision in bge_references if decision not in extracted_and_found]
        self.logger.info(f"{len(new_list)} references were extracted but not found")


if __name__ == '__main__':
    config = get_config()

    bge_criticality_dataset_creator = BgeCriticalityDatasetCreator(config)
    bge_criticality_dataset_creator.get_dataset('text', 'de', False)
