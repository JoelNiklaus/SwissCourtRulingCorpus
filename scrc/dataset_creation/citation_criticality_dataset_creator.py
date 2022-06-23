from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config
import numpy as np
from tqdm import tqdm
from pandarallel import pandarallel
import itertools
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from scrc.data_classes.ruling_citation import RulingCitation
import pandas as pd



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


class CitationCriticalityDatasetCreator(DatasetCreator):
    """Creates a dataset containing bger cases and sets for each case a criticality label, based how often the case
    was cited in bger cases"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.debug = True
        self.split_type = "date-stratified"
        # Todo check if names for each criticality creator should be unique
        self.dataset_name = "criticality_prediction"
        self.feature_cols = ['text']  # ['facts', 'considerations', 'text']

        self.load_rulings()

        # TODO what is this code doing?
        pandarallel.initialize(progress_bar=True)
        tqdm.pandas()

    def get_dataset(self, feature_col, lang, save_reports):
        """get all required data: all bge and bger cases and label bger cases"""

        df = self.get_df(self.get_engine(self.db_scrc), feature_col, 'citations', lang, save_reports)
        df = self.set_citation_criticality_label(df, feature_col, lang)

        # TODO need to drop something else?
        # df = df.drop(['citations', 'counter', 'rulings'], axis=1)
        # TODO rename neccessarry?
        # df = df.rename(columns={feature_col: "text"})  # normalize column names
        labels, _ = list(np.unique(np.hstack(df.citation_label), return_index=True))
        return df, labels

    def set_citation_criticality_label(self, df, feature_col, lang):
        """set for each bger ruling a label critical or non-critical depending on how often they
                were cited in other cases """
        # Include all bger rulings
        # get a list of number of citations of bge which were found in other rulings
        # define a minimum amount of citations needed to define a ruling as critical
        # 1. languages are different -> different datasets
        self.logger.info(f"Processing labeling of citation_criticality")

        # get a dict where for each bge ruling is counted how often it was cited by other rulings
        type_corpus_frequency = self.process_citation("rulings", df, lang)

        # create list of critical cases
        critical_cases = set()
        for file_number, citations_amount in type_corpus_frequency.items():
            if citations_amount >= 1:
                critical_cases.add(str(file_number))

        # set labels
        file_number_match = df.file_number.astype(str).isin(critical_cases)
        critical_df = df[file_number_match]
        critical_df['citation_label'] = 'critical'
        non_critical_df = df[~file_number_match]
        non_critical_df['citation_label'] = 'non-critical'
        return critical_df.append(non_critical_df)

    def process_citation(self, cit_type, df, lang):
        """find for each bge all citations in other bger"""
        self.logger.info(f"Processing the {cit_type} citations.")
        # TODO put used methods of Doc2DocIRDatasetCreator into abstract into parent DatasetCreator
        df[cit_type] = df.citations.parallel_apply(self.get_citations, type=cit_type, lang=lang)

        # we cannot use the ones which have no citations
        # because we drop everything we lose some data, but it is easier, because we know that all the entries rulings citations
        # reset the index so that we don't get out of bounds errors
        df = df.dropna(subset=[cit_type]).reset_index(drop=True)

        self.logger.info(f"Building the term-frequency matrix.")
        corpus = df[cit_type].tolist()
        vocabulary = sorted(list(set(itertools.chain(*corpus))))
        df[f"counter"] = df[cit_type].apply(lambda x: dict(Counter(x)))
        tf = self.build_tf_matrix(corpus, vocabulary, df)

        self.logger.info(f"Calculating the inverse document frequency scores.")
        tf_idf = TfidfTransformer().fit_transform(tf)

        self.logger.info("Computing relevance scores for documents in collection")
        # x.name retrieves the row index
        relevance_lambda = lambda x: {cit: self.compute_relevance_score(cit, tf_idf[x.name, vocabulary.index(cit)])
                                      for cit, _ in x[f"counter"].items()}
        df[f"{cit_type}_relevances"] = df.parallel_apply(relevance_lambda, axis=1)

        # counts how often a ruling is cited
        # TODO check dict if it is like {'2a 234/2017' : 2 , '1b 1234/2009 : 2}
        # assert no entry with 0 exists
        type_corpus_frequencies = dict(Counter(itertools.chain(*df[cit_type].tolist())))
        return type_corpus_frequencies

    def get_citations(self, citations, type, lang):
        cits = []
        for citation in citations[type]:
            cit = citation['text']
            cit = ' '.join(cit.split())  # remove multiple whitespaces inside
            try:
                if type == "rulings":
                    type_cit = RulingCitation(cit, lang)
                else:
                    raise ValueError("type must be either 'rulings' or 'laws'")
            except ValueError as ve:
                self.logger.debug(ve)
                continue
            # only actually include ruling citations that we can find in our corpus
            if type == "laws" or (type == "rulings" and str(type_cit) in self.available_bges):
                cits.append(type_cit)
        if cits:  # only return something if we actually have citations
            return cits

    @staticmethod
    def compute_relevance_score(cit, tf_idf_score):
        """
        Computes a relevance score for the citation between 0 and 1
        """
        return tf_idf_score

    @staticmethod
    def build_tf_matrix(corpus, vocabulary, type_df):
        # build term frequency matrix
        tf = np.zeros((len(corpus), len(vocabulary)))  # term frequencies: number of documents x number of citations

        def fill_tf_matrix(x):
            for cit, count in x[f"counter"].items():
                tf[x.name, vocabulary.index(cit)] = count

        type_df.progress_apply(fill_tf_matrix, axis=1)

        # TODO can be deleted
        # for doc_index, counter in tqdm(type_counter.iteritems(), total=len(corpus)):
        #    for cit, count in counter.items():
        #        cit_index = vocabulary.index(cit)
        #        tf[doc_index][cit_index] = count
        return tf

    def load_rulings(self):
        self.logger.info(f"Loading reference rulings")
        self.available_bges = set()  # use set instead of list for faster lookup
        rulings = pd.DataFrame()
        for lang in self.languages:
            # also for debugging we want the full list
            cols = "language, canton, date, file_number, html_url, text"
            df = next(self.select(self.get_engine(self.db_scrc), lang,
                                  columns=cols, where="spider = 'CH_BGE'", chunksize=self.real_chunksize))
            df.date = pd.to_datetime(df.date)
            assert len(df.index) == len(df.file_number.unique())  # there should not be any duplicates
            self.available_bges.update(df.file_number.tolist())
            rulings = pd.concat([rulings, df], ignore_index=True)  # one rulings df for all languages

        self.logger.info(f"Found {len(self.available_bges)} rulings")

if __name__ == '__main__':
    config = get_config()

    citation_criticality_dataset_creator = CitationCriticalityDatasetCreator(config)
    citation_criticality_dataset_creator.create_dataset(sub_datasets=False, kaggle=False, huggingface=False, save_reports=False)
