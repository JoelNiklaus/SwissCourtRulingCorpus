import ast
import pandas as pd
import itertools
import numpy as np

from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.data_classes.ruling_citation import RulingCitation
from root import ROOT_DIR
from pathlib import Path
from scrc.utils.main_utils import get_config
from scrc.utils.log_utils import get_logger
from collections import Counter


"""
Dataset to be created:
- contains supreme court cases  
- cols = year, legal_area, origin_region, origin_canton, considerations, facts, language, citation_label, bge_label
- only cases where feature_col =(facts, considerations) text has good length
- Dataset Split description:
    - train 70%
    - val 10%
    - test 20%
    - To consider: shuffle data or order by date?
Set Labels
    - BGE criticality
        - get list of extrcted bger_file_numbers
        - get all bger whose file numbers were extracted by bge_reference_extractor
        - set label critical for those found bger
    - citation criticality
        - count citations of bge in bger cases
        - create list of bge that were cited OPTIONAL: differ between amount of citations
        - get all bger_file_number for bge with reference extractor
        - set label critical for those found bger
        - OPTIONAL: distinct between citations depending on when they got cited
        ATTENTION citations criticality is a subset of bge criticality since only bge get cited
Error sources:
    - Regex cannot find correct file number in header
    - Regex found a wrong file
"""


class CriticalityDatasetCreator(DatasetCreator):
    """
    Creates a dataset containing bger cases and sets for each case a criticality label, based if the case was published
    as bge or not.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.debug = True
        self.split_type = "date-stratified"
        self.dataset_name = "criticality_prediction"
        self.feature_cols = ['facts', 'considerations']
        self.load_rulings()
        self.extract_bge_references()
        self.CITATION_COUNT_MIN = 1

    def get_dataset(self, feature_col, save_reports):
        """
        get of all bger cases and set bge_label and citation_label
        :param feature_col:     specify sections to add as column
        :param save_reports:    whether or not to compute and save reports
        :return:                dataframe and list of labels
        """
        df = self.get_df(self.get_engine(self.db_scrc), feature_col)
        self.logger.info(f"There are a total of {len(df.index)} bger cases.")
        bge_list = self.get_bge_criticality_list()
        df = self.set_criticality_label(df, bge_list, 'bge_label')
        criticality_list = self.get_citations_criticality_list(df)
        df = self.set_criticality_label(df, criticality_list, 'citation_label')
        df = self.filter_cases(df)
        bge_labels, _ = list(np.unique(np.hstack(df.bge_label), return_index=True))
        return df, bge_labels

    def filter_cases(self, df):
        """
        Filters given dataframe, to get rid of not needed data
        :param df:      dataframe of all bger cases
        :return:        filtered dataframe
        """
        columns = list(df.columns)
        columns.remove('year')
        columns.remove('legal_area')
        columns.remove('origin_region')
        columns.remove('origin_canton')
        columns.remove('bge_label')
        columns.remove('citation_label')
        columns.remove('lang')
        columns.remove('considerations')
        columns.remove('facts')
        df.drop(columns, axis=1, inplace=True)
        # TODO filter cases with too long / short input for model, maybe done in get_df
        return df

    def set_criticality_label(self, df, criticality_list, label):
        """
        Set label critical for cases in given list, else set label non-critical
        :param df:                  dataframe of all bger cases
        :param criticality_list:    list of bger_file_numbers which will be labeled critical
        :param label:               name of label in df
        :return:                    labeled dataframe
        """
        file_number_match = df.file_number.astype(str).isin(list(criticality_list))
        df.loc[file_number_match, label] = 'critical'
        df.loc[~file_number_match, label] = 'non-critical'
        self.logger.info(f"There are {len(df[file_number_match])} critical and {len(df[~file_number_match])} non-critical bger cases.")
        return df

    def load_rulings(self):
        """
        Load all bge cases and store in available_bges
        """
        self.logger.info(f"Loading reference rulings")
        self.available_bges = set()  # use set instead of list for faster lookup
        for lang in self.languages:
            df = next(self.select(self.get_engine(self.db_scrc), lang,
                                  columns="file_number", where="spider = 'CH_BGE'", chunksize=self.real_chunksize))
            assert len(df.index) == len(df.file_number.unique())  # there should not be any duplicates
            self.available_bges.update(df.file_number.tolist())
        self.logger.info(f"There are {len(self.available_bges)} BGE in total (also old or not referenced included).")

    def extract_bge_references(self):
        """
        Extract data from bge_refereces_found.txt
        :return:        dataframe containing bger_references and bge_file_numbers
        """
        # get dict of bge references with corresponding bge file name
        bge_references_file_path: Path = ROOT_DIR / 'data' / 'datasets' / "bge_references_found.txt"
        if not bge_references_file_path.exists():
            raise Exception("bge references need to be extracted first. Run bge_reference_extractor.")
        references = {}
        with bge_references_file_path.open("r") as f:
            for line in f:
                (bge_file_number, chamber, text) = line.split()
                bge_file_number = bge_file_number.split('_', 5)[3]
                bge_file_number = bge_file_number.replace('-', ' ')
                references[bge_file_number] = f"{chamber} {text}"
        self.logger.info(f"There are {len(list(references.keys()))} BGE where bger references could be extracted, {len(set(references.keys()))} unique.")
        # create data frame of dict
        data = {
            "bge_file_number": list(references.keys()),
            "bger_reference": list(references.values())
        }
        df = pd.DataFrame.from_dict(data)
        assert len(df.index) == len(df.bge_file_number.unique())
        self.references_df = df

    def get_bge_criticality_list(self):
        """
        create list of all bger_references
        :return:    return this list
        """
        self.logger.info(f"Processing labeling of bge_criticality")
        bge_list = self.references_df['bger_reference'].unique()
        return list(bge_list)

    def get_citations_criticality_list(self, df):
        """
        create a list of bger_references which were cited a specified amount
        :param df:      dataframe of all bger cases
        :return:        list of bger_references
        """
        self.logger.info(f"Processing labeling of citation_criticality")
        citations_df = self.count_citations_of_bger(df)
        # apply for each row a function which returns True if citation amount is bigger than ???
        critical_df = citations_df[citations_df['count'] >= self.CITATION_COUNT_MIN]
        self.logger.info(f"There were {len(critical_df.index)} cases found in extracted bger references which were cited at leat {self.CITATION_COUNT_MIN} times.")
        # create list of bge file numbers
        critical_bger_list = critical_df['bger_reference'].tolist()
        return critical_bger_list

    def count_citations_of_bger(self, df):
        """
        Count for bge amount of citations in other bger
        :param df:      dataframe of all bger cases
        :return:        dataframe containing columns 'bger_reference', 'bge_file_number' and 'count'
        """

        citation_frequencies_df = self.calculate_citation_frequencies(df)
        self.logger.info(f"Citation Criticality: There were {len(citation_frequencies_df.index)} unique bge cited")

        citation_count_df = self.references_df
        # iterate over unique count values
        a = citation_frequencies_df['count'].unique()
        for i in a:
            # set count in citation_count_df where citation_frequencies_df has matching reference and count value
            temp_freq_df = citation_frequencies_df[citation_frequencies_df['count'] == i]
            file_number_match = citation_count_df.bge_file_number.astype(str).isin(temp_freq_df['bge_file_number'].astype("string").tolist())
            citation_count_df.loc[file_number_match, 'count'] = int(i)
        return citation_count_df

    def calculate_citation_frequencies(self, df):
        """
        calculate how often a bge was cited in given bger cases
        :param df:        dataframe of all bger cases
        :return:          dataframe containing columns: 'bge_file_number' and 'count'
        """
        df = self.process_citation(df)
        self.logger.info(f"Building the term-frequency matrix.")
        df[f"counter"] = df['ruling_citation'].apply(lambda x: dict(Counter(x)))

        # counts how often a ruling is cited
        # assert no entry with 0 exists
        type_corpus_frequencies = dict(Counter(itertools.chain(*df['ruling_citation'].tolist())))
        data = {
            "bge_file_number": list(type_corpus_frequencies.keys()),
            "count": list(type_corpus_frequencies.values())
        }
        return pd.DataFrame.from_dict(data)

    def process_citation(self, df):
        """
        extract for each bger all ruling citations
        :param df:        dataframe of all bger cases
        :return:          dataframe with additional column 'ruling_citation'
        """
        self.logger.info(f"Processing to extract for each bger all citations which were found.")
        df['ruling_citation'] = df.citations.apply(self.get_citations)
        # get rid of reset the index so that we don't get out of bounds errors
        return df.dropna(subset=['ruling_citation']).reset_index(drop=True)

    def get_citations(self, citations_as_string):
        """
        extract for each bger all ruling citations
        :param citations_as_string:         citations how they were found in text of bger
        :return:                            dataframe with additional column 'ruling_citation'
        """
        cits = []
        try:
            citations = ast.literal_eval(citations_as_string)  # parse dict string to dict again
            for citation in citations:
                try:
                    cit = citation['text']
                    citation_type = citation['name']
                    cit = ' '.join(cit.split())  # remove multiple whitespaces inside
                    if citation_type == "ruling":
                        cited_file = self.get_file_number(cit)
                        cits.append(cited_file)
                    elif citation_type == "law":
                        pass
                    else:
                        raise ValueError("type of citation must be 'rulings' or 'law")
                except ValueError as ve:
                    self.logger.info("citation has invalid syntax")
                    self.logger.info(citation)
                    continue
        except ValueError as ve:
            self.logger.info("citations could not be extracted to dict")
            self.logger.info(citations_as_string)
        if cits:  # only return something if we actually have citations
            return cits

    def get_file_number(self, citation):
        """
        find for each citation string the matching citation from the start of bge (first page)
        :param citation:         citation as string as found in text
        :return:                 RulingCitation always in German
        """
        # handle citation always in German
        found_citation = RulingCitation(citation, 'de')
        if str(found_citation) in self.available_bges:
            return found_citation
        else:
            # find closest bge with smaller page_number
            year = found_citation.year
            volume = found_citation.volume
            page_number = found_citation.page_number
            new_page_number = -1
            for match in self.available_bges:
                if f"BGE {year} {volume}" in match:
                    tmp = RulingCitation(match, 'de')
                    if new_page_number < tmp.page_number <= page_number:
                        new_page_number = tmp.page_number
            return RulingCitation(f"{year} {volume} {new_page_number}", 'de')

    def check_extracted_bger_references(self):
        bge_references_file_path: Path = ROOT_DIR / 'data' / 'datasets' / "bge_references_found.txt"
        if not bge_references_file_path.exists():
            raise Exception("bge references need to be extracted first. Run bge_reference_extractor.")
        references = {}
        i = 0
        j = 0
        with bge_references_file_path.open("r") as f:
            for line in f:
                (bge_file_number, chamber, text) = line.split()
                bge_file_number = bge_file_number.split('_', 5)[3]
                year = int(bge_file_number.split('-')[1])
                if year < 133:
                    j = j+1
                bge_file_number = bge_file_number.replace('-', ' ')
                if bge_file_number in list(references.keys()):
                    i = i + 1
                    if bge_file_number.__eq__(str(references[bge_file_number])):
                        print(
                            f"Already exists {bge_file_number} old: {references[bge_file_number]} new {chamber} {text}")
                else:
                    references[bge_file_number] = f"{chamber} {text}"
        self.logger.info(f"there were {i} cases where the same file_number was extracted")
        self.logger.info(f"there are {j} cases from before 2007.")
        self.logger.info(
            f"There are {len(references.keys())} BGE where bger references could be extracted, {len(set(references.keys()))} unique.")

        bge_references_file_path: Path = ROOT_DIR / 'data' / 'datasets' / "bge_not_extracted.txt"
        references = {}
        i = 0
        with bge_references_file_path.open("r") as f:
            for line in f:
                year = int(line.split('-', 5)[1])
                if year >= 133:
                    i = i+1
        self.logger.info(f"in total there are {i} cases which are not extracted which are from 2007 or 2008.")
        # TODO check if citations get extracted and correctly referenced


if __name__ == '__main__':
    config = get_config()

    criticality_dataset_creator = CriticalityDatasetCreator(config)
    criticality_dataset_creator.create_dataset(sub_datasets=False, kaggle=False, huggingface=True, save_reports=False)
    # criticality_dataset_creator.check_extracted_bger_references()

