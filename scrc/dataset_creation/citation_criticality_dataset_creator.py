from scrc.data_classes.ruling_citation import RulingCitation
from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.utils.log_utils import get_logger
import pandas as pd

from scrc.utils.main_utils import get_config
from scrc.dataset_creation.criticality_dataset_creator import CriticalityDatasetCreator


class CitationCriticalityDatasetCreator(CriticalityDatasetCreator):
    """
    Creates a dataset with criticality labels based on the definition of bge criticality
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # self.dataset_name = "criticality_prediction"
        # self.feature_cols = ['text']  # ['facts', 'considerations', 'text']

    # set criticality labels
    def get_labeled_data(self, bger_df, bge_df):
        self.logger.info(f"Processing labeling of bge_criticality")

        # Include all bger rulings whose file_number can be found in the header of a bge
        # get a list of number of citations of bge which were found in other rulings
        # define a minimum amount of citations needed to define a ruling as critical
        # 1. Regex cannot find correct file number in header
        # 2. languages are different -> different datasets

        # TODO get all citations in a bger ruling

        # TODO count amount of citations for one bge in all bger rulings

        # TODO safe this number in the database?

        """
        criticality_score_match = bger_df.citations >= 10
        critical_df = bger_df[criticality_score_match]
        critical_df['label'] = 'critical'
        non_critical_df = bger_df[~criticality_score_match]
        non_critical_df['label'] = 'non-critical'
        return critical_df.append(non_critical_df)
        """
        return bger_df

    def process_citation_for_bger(self, df, lang):
        self.logger.info(f"Processing the ruling citations.")
        df['citations'] = df.citations.parallel_apply(self.get_citations, lang=lang)

        # we cannot use the ones which have no citations
        # because we drop everything we lose some data, but it is easier, because we know that all the entries have both laws citations and rulings citations
        # reset the index so that we don't get out of bounds errors
        df = df.dropna(subset=['citation']).reset_index(drop=True)

        return df

    def get_citations(self, citations, lang):
        cits = []
        cit = citations['text']
        cit = ' '.join(cit.split())  # remove multiple whitespaces inside
        type_cit = RulingCitation(cit, lang)
        # only actually include ruling citations that we can find in our corpus
        cits.append(type_cit)
        if cits:  # only return something if we actually have citations
            return cits


if __name__ == '__main__':
    config = get_config()

    citation_criticality_dataset_creator = CitationCriticalityDatasetCreator(config)
    citation_criticality_dataset_creator.create_dataset()
