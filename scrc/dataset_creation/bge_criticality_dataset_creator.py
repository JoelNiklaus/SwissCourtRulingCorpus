from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.dataset_creation.criticality_dataset_creator import CriticalityDatasetCreator
from root import ROOT_DIR
from pathlib import Path
from scrc.utils.main_utils import get_config
from scrc.utils.log_utils import get_logger
import numpy as np
from tqdm import tqdm
from pandarallel import pandarallel
from scrc.utils.sql_select_utils import get_legal_area, legal_areas, get_region, \
    select_paragraphs_with_decision_and_meta_data, where_string_spider

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
"""


class BgeCriticalityDatasetCreator(DatasetCreator):
    """
    Creates a dataset containing bger cases and sets for each case a criticality label, based if the case was published
    as bge or not.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.debug = True
        self.split_type = "date-stratified"
        # Todo check if names for each criticality creator should be unique
        self.dataset_name = "criticality_prediction"
        self.feature_cols = ['text']  # ['facts', 'considerations', 'text']

        # TODO what is this code doing?
        pandarallel.initialize(progress_bar=True)
        tqdm.pandas()

    def get_dataset(self, feature_col, lang, save_reports):
        """get all bger cases and set labels"""

        # TODO change how data is received, why is it not working?
        # df = self.get_df(self.get_engine(self.db_scrc), feature_col, 'citations', lang, save_reports)
        engine = self.get_engine(self.db_scrc)
        df = self.query_bger(feature_col, engine, lang)
        df['legal_area'] = 'Strafrecht'
        df['origin_region'] = 'Aargau'
        df['origin_canton'] = 'Bern'
        df = self.set_bge_criticality_label(df)

        # TODO need to drop something else?
        # df = df.drop(['citations', 'counter', 'rulings'], axis=1)
        # TODO rename neccessarry?
        df = df.rename(columns={feature_col: "text"})  # normalize column names
        df = df.rename(columns={'bge_label': "label"})
        labels, _ = list(np.unique(np.hstack(df.label), return_index=True))
        return df, labels

    def set_bge_criticality_label(self, df):
        """set for each bger ruling a label critical or non-critical depending on whether their
        file number was extracted in a bge"""
        # Include all bger rulings whose file_number can be found in the header of a bge
        # error sources:
        # 1. Regex cannot find correct file number in header
        self.logger.info(f"Processing labeling of bge_criticality")

        bge_references_file_path: Path = ROOT_DIR / 'data' / 'progress' / "bge_references_found.txt"
        if not bge_references_file_path.exists():
            raise Exception("bge references need to be extracted first. Run bge_reference_extractor.")
        bge_references = bge_references_file_path.read_text().strip().split("\n")
        file_number_match = df.file_number.astype(str).isin(list(bge_references))
        critical_df = df[file_number_match]
        critical_df['bge_label'] = 'critical'
        non_critical_df = df[~file_number_match]
        non_critical_df['bge_label'] = 'non-critical'
        self.logger.info(f"# critical decisions: {len(critical_df.index)}")
        self.logger.info(f"# non-critical decisions: {len(non_critical_df.index)}")
        self.calculate_label_coverage(bge_references, file_number_match, critical_df, df)
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

    def query_bger(self, feature_col, engine, lang):
        """get all bger form database"""
        # TODO which columns are needed
        columns = ['id', 'chamber', 'date', 'extract(year from date) as year', f'{feature_col}', 'file_name', 'file_number']
        try:
            bger_df = next(self.select(engine, lang,
                                      columns=",".join(columns),
                                      where="court = 'CH_BGer'",
                                      order_by="date",
                                      chunksize=self.get_chunksize()))
        except StopIteration:
            raise ValueError("No bger rulings found")
        # get rid of all dublicated cases
        # TODO improve this
        bger_df = bger_df.dropna(subset=['date', 'id'])
        self.logger.info(f"Found {len(bger_df.index)} supreme bger rulings")
        # TODO filter cases with too long / short input for model
        return bger_df


if __name__ == '__main__':
    config = get_config()

    bge_criticality_dataset_creator = BgeCriticalityDatasetCreator(config)
    bge_criticality_dataset_creator.create_dataset(sub_datasets=False, kaggle=False, huggingface=True, save_reports=False)

