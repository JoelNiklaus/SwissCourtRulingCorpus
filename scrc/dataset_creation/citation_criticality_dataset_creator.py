from scrc.data_classes.ruling_citation import RulingCitation
from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.utils.log_utils import get_logger
import pandas as pd

import itertools
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer

from scrc.utils.main_utils import get_config
from scrc.dataset_creation.criticality_dataset_creator import CriticalityDatasetCreator
from scrc.dataset_creation.doc2doc_ir_dataset_creator import Doc2DocIRDatasetCreator

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
        # TODO take care of all languages
        type_corpus_frequency = self.process_citation_type("rulings", bger_df, 'de')
        self.logger.info(f"Got the processed frequencies")

        # Include all bger rulings
        # get a list of number of citations of bge which were found in other rulings
        # define a minimum amount of citations needed to define a ruling as critical
        # 1. languages are different -> different datasets

        """
        critical_case_list = list(str(type_corpus_frequency[file_number]) where type_corpus_frequency[score]>=1)
        case_cited = bger_df.file_number.astype(str).isin(critical_case_list)
        critical_df = bger_df[case_cited]
        critical_df['label'] = 'critical'
        non_critical_df = bger_df[~criticality_score_match]
        non_critical_df['label'] = 'non-critical'
        return critical_df.append(non_critical_df)
        """
        return bger_df

    def process_citation_type(self, cit_type, df, lang):
        self.logger.info(f"Processing the {cit_type} citations.")
        df[cit_type] = df.citations.parallel_apply(Doc2DocIRDatasetCreator.get_citations, type=cit_type, lang=lang)

        # we cannot use the ones which have no citations
        # because we drop everything we lose some data, but it is easier, because we know that all the entries have both laws citations and rulings citations
        # reset the index so that we don't get out of bounds errors
        df = df.dropna(subset=[cit_type]).reset_index(drop=True)

        self.logger.info(f"Building the term-frequency matrix.")
        corpus = df[cit_type].tolist()
        vocabulary = sorted(list(set(itertools.chain(*corpus))))
        df[f"counter"] = df[cit_type].apply(lambda x: dict(Counter(x)))
        tf = Doc2DocIRDatasetCreator.build_tf_matrix(corpus, vocabulary, df)

        self.logger.info(f"Calculating the inverse document frequency scores.")
        tf_idf = TfidfTransformer().fit_transform(tf)

        self.logger.info("Computing relevance scores for documents in collection")
        # x.name retrieves the row index
        relevance_lambda = lambda x: {cit: self.compute_relevance_score(cit, tf_idf[x.name, vocabulary.index(cit)])
                                      for cit, _ in x[f"counter"].items()}
        df[f"{cit_type}_relevances"] = df.parallel_apply(relevance_lambda, axis=1)

        # counts how often a ruling is cited
        # check dict if it is like {'2a 234/2017' : 2 , '1b 1234/2009 : 2}
        # assert no entry with 0 exists
        type_corpus_frequencies = dict(Counter(itertools.chain(*df[cit_type].tolist())))
        return type_corpus_frequencies


    @staticmethod
    def compute_relevance_score(cit, tf_idf_score):
        """
        Computes a relevance score for the citation between 0 and 1
        """
        return tf_idf_score


if __name__ == '__main__':
    config = get_config()

    citation_criticality_dataset_creator = CitationCriticalityDatasetCreator(config)
    citation_criticality_dataset_creator.create_dataset()
