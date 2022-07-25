from abc import ABC, abstractmethod
import itertools
from collections import Counter

from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm
from scrc.data_classes.law_citation import LawCitation
from scrc.data_classes.ruling_citation import RulingCitation
from scrc.dataset_creation.dataset_creator import DatasetCreator
import numpy as np
import pandas as pd
from pandarallel import pandarallel
import re
import json


class CitationDatasetCreator(ABC, DatasetCreator):
    """Abstract Base Class to unify methods for citatition_criticality_dataset_creator and
    doc2doc_ir_dataset_creator"""

    def __init__(self, config: dict):
        super().__init__(config)

        # self.dataset_folder = self.create_dir(self.datasets_subdir, self.dataset_name)

        pandarallel.initialize(progress_bar=True)
        tqdm.pandas()

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
        rulings_path = self.dataset_folder / "rulings.jsonl"
        if not rulings_path.exists():
            rulings.to_json(rulings_path, orient="records", lines=True, date_format="iso")

    def process_citation_type(self, cit_type, df, lang):
        self.logger.info(f"Processing the {cit_type} citations.")
        df[cit_type] = df.citations.parallel_apply(self.get_citations, type=cit_type, lang=lang)

        # we cannot use the ones which have no citations
        # because we drop everything we lose some data, but it is easier, because we know that all the entries have both laws citations and rulings citations
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

        # TODO move this to the plot_custom function so that we can save it for all languages
        self.logger.info("Saving graphs about the dataset.")
        type_corpus_frequencies = dict(Counter(itertools.chain(*df[cit_type].tolist())))
        self.save_citations(type_corpus_frequencies, cit_type)
        return df

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

    def save_citations(self, citation_frequencies: dict, name: str, top_n=25):
        figure_path = self.dataset_folder / f"{name}_citations.png"
        csv_path = self.dataset_folder / f"{name}_citations.csv"

        citations = pd.DataFrame.from_dict(citation_frequencies, orient="index", columns=["frequency"])
        citations = citations.sort_values(by=['frequency'], ascending=False)
        citations.to_csv(csv_path)

        ax = citations[:top_n].plot.bar(use_index=True, y='frequency', rot=90)
        ax.get_figure().savefig(figure_path, bbox_inches="tight")

    def get_citations(self, citations_as_string, lang):
        citations = list(eval(citations_as_string))
        data = json.load(citations_as_string)
        cits = []
        for citation in citations:
            try:
                cit = citation['text']
                type = citation['name']
                cit = ' '.join(cit.split())  # remove multiple whitespaces inside
                try:
                    if type == "ruling":
                        file_number = self.get_file_number(cit, lang)
                        type_cit = RulingCitation(file_number, lang)
                        cits.append(type_cit)
                    elif type == "law":
                        pass
                    else:
                        raise ValueError("type must be 'rulings'")
                except ValueError as ve:
                    self.logger.debug(ve)
                    continue
            except ValueError as ve:
                self.logger.info(citation)
        if cits:  # only return something if we actually have citations
            return cits

    @staticmethod
    def compute_relevance_score(cit, tf_idf_score):
        """
        Computes a relevance score for the citation between 0 and 1
        Algorithm:
        1. tf_idf score for citation: frequency of citation in the considerations (==> main question) *
         inverse frequency of citation in corpus (over all splits for code simplicity) (==> most specific citation)
        2. maybe consider criticality score from Ronja
        :return:
        # Proxy für relevanz score:
        #   - tf für zitierungen: häufigkeit des Zitats in Erwägungen
        #   - idf für zitierungen: wenn Zitat im Korpus häufig vorkommt hat es tiefe Relevanz ==> neuere BGEs kommen weniger häufig vor und sind somit wichtiger
        #   - neuere Urteile sind relevanter (ältere Urteile sind evtl. überholt oder nicht mehr relevant), aber es hängt von der Frage ab (pro rechtlicher Frage wäre das neuere BGE relevanter)
        #   - BGE wichtiger als andere ==> nur BGEs nehmen
        Future work: investigate IR per legal question
        """
        # TODO add Ronja's criticality score
        return tf_idf_score

    def get_file_number(self, type_cit, lang):
        if str(type_cit) in self.available_bges:
            return str(type_cit)
        else:
            bge = RulingCitation(type_cit, lang)
            year = bge.year
            volume = bge.volume
            page_number = bge.page_number
            origin_page_number = -1
            if f"BGE {year} {volume}" == "BGE 147 III":
                print(bge)
            for match in self.available_bges:
                if f"BGE {year} {volume}" in match:
                    bge = RulingCitation(match, lang)
                    if origin_page_number < bge.page_number <= page_number:
                        origin_page_number = bge.page_number
            return f"BGE {year} {volume} {origin_page_number}"
