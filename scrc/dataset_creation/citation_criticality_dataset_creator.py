from scrc.dataset_creation.citation_dataset_creator import CitationDatasetCreator
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config
import numpy as np
import itertools
from collections import Counter

from sklearn.feature_extraction.text import TfidfTransformer




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
    - criticality based on ruling citations
        - count citations of bge in bger cases using methods of Doc2Doc
        - set label critical if number of citations found for a bge
        - only bge get citated -> only part of bge will be critical
        - OPTIONAL: distinct between citations depending on when they got cited
Check distribution of data sets
    - distribution among languages
    - distribution among legal areas
    - distribution among cantons
    - is there bias detectable?
"""


class CitationCriticalityDatasetCreator(CitationDatasetCreator):
    """Creates a dataset containing bger cases and sets for each case a criticality label, based how often the case
    was cited in bger cases"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.debug = True
        self.split_type = "date-stratified"
        self.dataset_name = "citation_criticality_prediction"
        self.feature_cols = ['text']  # ['facts', 'considerations', 'text']
        self.dataset_folder = self.create_dir(self.datasets_subdir, self.dataset_name)
        self.load_rulings()

    def get_dataset(self, feature_col, save_reports):
        """get all required data: all bge and bger cases and label bger cases"""

        df = self.get_df(self.get_engine(self.db_scrc), feature_col, 'citations', save_reports)
        df = self.count_citations_per_ruling(df)
        df = self.set_citation_criticality_label(df, feature_col)
        # rename columns
        df = df.rename(columns={'citation_label': "label"})  # normalize column names
        labels, _ = list(np.unique(np.hstack(df.citation_label), return_index=True))
        return df, labels

    def set_citation_criticality_label(self, df, feature_col):
        """set for each bger ruling a label critical or non-critical depending on how often they
                were cited in other cases """
        self.logger.info(f"Processing labeling of citation_criticality")
        # apply for each row a function which returns true if citation amount is bigger than ???

        def critical(x):
            if int(x) > 1:
                return 'critical'
            else:
                return 'non-critical'

        df['citation-label'] = df['citation_count'].apply(critical)
        return df

    def process_citation(self, df):
        """find for each bge all citations in other bger"""
        self.logger.info(f"Processing the ruling citations.")
        # TODO remove lang from methods
        df['cit_type'] = df.citations.parallel_apply(self.get_citations, type='rulings', lang='de')

        # we cannot use the ones which have no citations
        # because we drop everything we lose some data, but it is easier, because we know that all the entries rulings citations
        # reset the index so that we don't get out of bounds errors
        df = df.dropna(subset=['cit_type']).reset_index(drop=True)

        self.logger.info(f"Building the term-frequency matrix.")
        df[f"counter"] = df['cit_type'].apply(lambda x: dict(Counter(x)))

        # counts how often a ruling is cited
        # assert no entry with 0 exists
        type_corpus_frequencies = dict(Counter(itertools.chain(*df['cit_type'].tolist())))
        return type_corpus_frequencies

    def count_citations_per_ruling(self, df):
        df['citation_count'] = 0
        # get a dict where for each bge ruling is counted how often it was cited by other rulings
        # dict is like {'2a 234/2017' : 2 , '1b 1234/2009 : 2}
        type_corpus_frequency = self.process_citation("rulings", df)
        # TODO add for each case amount of citations using file_number
        return df


if __name__ == '__main__':
    config = get_config()

    citation_criticality_dataset_creator = CitationCriticalityDatasetCreator(config)
    citation_criticality_dataset_creator.create_dataset(sub_datasets=False, kaggle=False, huggingface=False, save_reports=False)
