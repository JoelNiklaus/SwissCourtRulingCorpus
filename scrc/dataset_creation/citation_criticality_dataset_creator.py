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
        """set for each bger ruling a label critical or non-critical depending on how often a case was cited in another
            case or not.
            One should think about where citations can be found. Only in bger or also in cantons.
        """
        self.logger.info(f"Processing labeling of citation_criticality")
        self.logger.info(f"# there are {len(bger_df.index)} bger decisions")
        self.logger.info(f"# there are {len(bge_df.index)} bge decisions")
        self.logger.info(f"Processing labeling of citation_criticality")

        # Include all bger rulings
        # get a list of number of citations of bge which were found in other rulings
        # define a minimum amount of citations needed to define a ruling as critical
        # 1. languages are different -> different datasets

        # TODO take care of all languages
        # get a dict where for each bge ruling is counted how often it was cited by other rulings
        type_corpus_frequency = self.process_citation("rulings", bger_df, 'de')

        critical_cases = set()
        for file_number, citations_amount in type_corpus_frequency.items():
            if citations_amount >= 1:
                critical_cases.add(str(file_number))

        file_number_match = bger_df.file_number.astype(str).isin(critical_cases)
        critical_df = bger_df[file_number_match]
        critical_df['citation_label'] = 'critical'
        non_critical_df = bger_df[~file_number_match]
        non_critical_df['citation_label'] = 'non-critical'
        return critical_df.append(non_critical_df)

    def process_citation(self, cit_type, df, lang):
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
    citation_criticality_dataset_creator.get_dataset('text', 'de', False)
