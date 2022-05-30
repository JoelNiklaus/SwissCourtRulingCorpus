from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.utils.log_utils import get_logger
import pandas as pd
from abc import ABC, abstractmethod

from scrc.utils.main_utils import get_config

"""
Datasets to be created:
- BGE
    Contains all BGE since ? 
    cols = language, canton, date, file-number, text
- BGer
    contains all bger since 
    cols = language, canton, date, file-number, text
Set Labels
    - criticality based on BGE
        - get all bger whose file numbers appear in a bge header
        - use of "bge_reference_extractor"
        - set label critical for those found bger
    - criticality based on ruling citations
        - set label according number of citations found for a bge
        - critical if score more than ??? citations are found
        - distinct between links that are 10 years old and those which are newer
        - set of critical cases using citations is subset of criticality based on BGE
    - criticality based on published "Medienmitteilungen"
        - 
"""


class CriticalityDatasetCreator(ABC, DatasetCreator):
    """Abstract Base Class used by criticality dataset creators to unify their behaviour"""

    # TODO filter out cases where facts or other input for training model is too short
    # - what is used as input?
    # - what's the input length?
    # TODO filter only supreme court cases
    # - are there any constraints? time, legal area, ...
    # TODO Criticality definition 1
    # - Filter all cases that were published with abbreviation BGE
    # - Check if one can find matching bger cases for all BGE cases
    # - make sure no case is found twice
    # - define all BGE cases as criticalBGE
    # TODO Criticality definition 2
    # - get data set of newspaper occurrences
    # - check if one needs to filter certain cases or if all can be used
    # - get case for a occurrence in newspaper
    # - define all cases with occurrence as criticalNEWS
    # TODO Criticality definition 3
    # - get data set of links / references
    # - define all cases that were referenced to as criticalLINK
    # TODO Check distribution of data sets
    # - distribution among languages
    # - distribution among legal areas
    # - distribution among cantons
    # - is there bias detectable?

    @abstractmethod
    def get_labeled_data(self, bger_df: pd.DataFrame, bge_df: pd.DataFrame):
        """Returns the labeled data and labels"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.debug = False
        self.split_type = "date-stratified"

        #Todo check if names for each criticality creator should be unique
        self.dataset_name = "criticality_prediction"
        self.feature_cols = ['text']  # ['facts', 'considerations', 'text']

        # self.with_partials = False
        # self.with_write_off = False
        # self.with_unification = False
        # self.with_inadmissible = False
        # self.make_single_label = True

    def get_dataset(self, feature_col, lang, save_reports):
        # create engine
        engine = self.get_engine(self.db_scrc)
        # get bge rulings
        bge_df = self.query_bge(feature_col, engine, lang)
        # get bger rulings
        bger_df = self.query_bger(feature_col, engine, lang)
        # set criticality label
        bger_criticality_df = self.get_labeled_data(bger_df, bge_df)
        labels = ['non-critical', 'critical']
        return bger_criticality_df, labels

    # get all bger
    def query_bger(self, feature_col, engine, lang):
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
        return bger_df

    # get all bge
    def query_bge(self, feature_col, engine, lang):
        # TODO which columns are needed
        columns = ['id', 'chamber', 'date', 'extract(year from date) as year', f'{feature_col}', 'file_name', 'file_number']
        try:
            bge_df = next(self.select(engine, lang,
                                                columns=",".join(columns),
                                                where="spider = 'CH_BGE'",
                                                order_by="date",
                                                chunksize=self.get_chunksize()))
        except StopIteration:
            raise ValueError("No bge rulings found")
        # get rid of all dublicated cases
        # TODO improve this
        bge_df = bge_df.dropna(subset=['date', 'id'])
        self.logger.info(f"Found {len(bge_df.index)} supreme bge rulings")
        return bge_df
