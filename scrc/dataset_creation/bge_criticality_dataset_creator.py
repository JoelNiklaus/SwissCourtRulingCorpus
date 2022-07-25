from scrc.dataset_creation.dataset_creator import DatasetCreator
from root import ROOT_DIR
from pathlib import Path
from scrc.utils.main_utils import get_config
from scrc.utils.log_utils import get_logger
import numpy as np

"""
Dataset to be created:
- contains supreme court cases  
- cols = feature_col and label
- only cases where feature_col text has good length
- Dataset description:
    - train.jsonl:
        contains only the train split of queries.jsonl (years 2000 - 2014)
    - val.jsonl:
        contains only the validation split of queries.jsonl (years 2015 - 2016)
    - test.jsonl:
        contains only the test split of queries.jsonl (years 2017 - 2021)
Set Labels
    - criticality based on BGE
        - get all bger whose file numbers were extracted by bge_reference_extractor
        - set label critical for those found bger
Check distribution of data sets
    - distribution among languages
    - distribution among legal areas
    - distribution among cantons
    - is there bias detectable?
    
Error sources:
    - Regex cannot find correct file number in header
    - Regex found a wrong file
"""


class BgeCriticalityDatasetCreator(DatasetCreator):
    """
    Creates a dataset containing bger cases and sets for each case a criticality label, based if the case was published
    as bge or not.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.debug = False
        self.split_type = "date-stratified"
        self.dataset_name = "bge_criticality_prediction"
        self.feature_cols = ['facts', 'considerations']

    def get_dataset(self, feature_col, save_reports):
        """get all bger cases and set labels"""
        df = self.get_df(self.get_engine(self.db_scrc), feature_col, 'not needed', 'not needed')
        df = self.set_bge_criticality_label(df)
        df = self.filter_cases(df, feature_col)
        labels, _ = list(np.unique(np.hstack(df.label), return_index=True))
        return df, labels

    def filter_cases(self, df, feature_col):
        # TODO filter cases with too long / short input for model, maybe done in get_df
        return df

    def set_bge_criticality_label(self, df):
        """set for each bger ruling a label critical or non-critical depending on whether their
        file number was extracted in a bge"""
        self.logger.info(f"Processing labeling of bge_criticality")

        bge_references_file_path: Path = ROOT_DIR / 'data' / 'progress' / "bge_references_found.txt"
        if not bge_references_file_path.exists():
            raise Exception("bge references need to be extracted first. Run bge_reference_extractor.")
        references = {}
        with bge_references_file_path.open("r") as f:
            for line in f:
                (bge_file_number, chamber, text) = line.split()
                # found bge_file_number with pattern like: CH_BGE_007_BGE-144-V-236_2018 convert to BGE 144 V 236
                bge_file_number = bge_file_number.split('_', 5)[3]
                bge_file_number = bge_file_number.replace('-', ' ')
                references[bge_file_number] = f"{chamber} {text}"
        bge_references = list(references.values())
        file_number_match = df.file_number.astype(str).isin(list(bge_references))
        critical_df = df[file_number_match]
        critical_df['bge_label'] = 'critical'
        non_critical_df = df[~file_number_match]
        non_critical_df['bge_label'] = 'non-critical'
        self.logger.info(f"# critical decisions: {len(critical_df.index)}")
        self.logger.info(f"# non-critical decisions: {len(non_critical_df.index)}")
        return critical_df.append(non_critical_df)


if __name__ == '__main__':
    config = get_config()

    bge_criticality_dataset_creator = BgeCriticalityDatasetCreator(config)
    bge_criticality_dataset_creator.create_dataset(sub_datasets=False, kaggle=False, huggingface=False, save_reports=False)


