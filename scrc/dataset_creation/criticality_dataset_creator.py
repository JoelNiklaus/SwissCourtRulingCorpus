import os.path
import math

import pandas as pd
import numpy as np
import datasets

import plotly.express as px
import matplotlib.pyplot as plt

from scrc.dataset_creation.dataset_creator import DatasetCreator
from pathlib import Path

from scrc.dataset_creation.report_creator import ReportCreator
from scrc.enums.section import Section
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

    def __init__(self, config: dict, debug: bool = False):
        super().__init__(config, debug)
        self.logger = get_logger(__name__)
        self.split_type = "date-stratified"
        self.dataset_name = "criticality_prediction"
        self.feature_cols = [Section.FACTS, Section.CONSIDERATIONS]
        self.available_bges = self.load_rulings()
        self.references_df = self.extract_bge_references()
        self.citation_amount = [100, 10, 1, 0]  # sorted, the highest number first!
        self.labels = ['bge_label', 'citation_label']
        self.count_all_cits = False
        self.first_year = 2001

    def extract_bge_references(self):
        """
        Extract data from bge_refereces_found.txt
        :return:        dataframe containing bger_references and bge_file_numbers
        """
        # get dict of bge references with corresponding bge file name
        bge_references_file_path: Path = self.get_dataset_folder() / "bge_references_found.txt"
        if not bge_references_file_path.exists():
            raise Exception("bge references need to be extracted first. Run bge_reference_extractor.")
        df = pd.DataFrame({'bge_file_number_long': [], 'bge_file_number_short': [], 'bger_reference': [], 'year': [],
                           'bge_chamber': []})
        with bge_references_file_path.open("r") as f:
            for line in f:
                (bge_file_number_long, bger_reference) = line.split()
                year = int(bge_file_number_long.split('-', 5)[1]) + 1874
                bge_chamber = bge_file_number_long.split('_', 5)[2]
                bge_file_number_short = bge_file_number_long.split('_')[3]
                s_row = pd.Series([bge_file_number_long, bge_file_number_short, bger_reference, year, bge_chamber],
                                  index=df.columns)
                df = df.append(s_row, ignore_index=True)
        self.logger.info(
            f"References_df: There are {len(df.index)} entries with {len(df.bge_file_number_long.unique())} unique bge_filenumbers and {len(df.bger_reference.unique())} unique bger_references.")
        folder = self.create_dir(self.get_dataset_folder(), f'reports/references')
        report_creator = ReportCreator(folder, self.debug)
        report_creator.report_citations_count(df, 'references')
        return df

    def prepare_dataset(self, save_reports, court_string):
        """
        get of all bger cases and set bge_label and citation_label
        :param save_reports:    whether or not to compute and save reports
        :param court_string:    specifies court
        :return:                dataframe and list of labels
        """
        data_to_load = {
            "section": True, "file": True, "file_number": True,
            "judgment": False, "citation": True, "lower_court": True
        }
        df = self.get_df(self.get_engine(self.db_scrc), data_to_load)
        a = len(df['file_number'].unique())
        self.logger.info(f"BGer: There are {len(df.index)} bger cases with {a} unique file numbers.")

        # bge criticality
        bge_list = self.get_bge_criticality_list()
        df = self.set_criticality_label(df, bge_list, 'bge_label')

        # citation criticality
        criticality_lists = self.get_citations_criticality_list(df, self.citation_amount)
        # to debug use these given lists so citation_label_{i} is sometimes critical
        # clist = ['6B_1045/2018']
        # blist = ['8C_309/2009', '2A_88/2003']
        # alist = ['8C_309/2009', '2A_88/2003', '1C_396/2017', '5A_960/2019']
        # criticality_lists = [clist, blist, alist, bge_list]
        df_list = []
        i = 0
        for critical_list in criticality_lists:
            if critical_list:
                df = self.set_criticality_label(df, critical_list, 'citation_label')
                critical_df = df[df['citation_label'] == 'critical']
                df = df[df['citation_label'] != 'critical']
                critical_df['citation_label'] = critical_df['citation_label'].map(
                    {'critical': f"critical-{self.citation_amount[i]}"}, na_action=None)
                if not critical_df.empty:
                    df_list.append(critical_df)
            i = i + 1
        df['citation_label'] = 'non-critical'
        df_list.append(df)
        df = pd.concat(df_list)

        bge_labels, _ = list(np.unique(np.hstack(df.bge_label), return_index=True))
        citation_labels, _ = list(np.unique(np.hstack(df.citation_label), return_index=True))
        label_list = [bge_labels, citation_labels]
        self.logger.info("finished criticality")
        return df, label_list

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
        critical_df = df[df[label] == 'critical']
        self.logger.info(
            f"{label}: There are {len(critical_df.index)} critical cases with {len(critical_df.file_number.unique())} unique file_numbers.")
        if label == "bge_label":
            df.loc[~file_number_match, label] = 'non-critical'
            # Check which file numbers could not been found for bge criticality
            labeled_critical_list = list(critical_df.file_number.unique())
            not_found_list = [item for item in criticality_list if item not in labeled_critical_list]
            folder = self.create_dir(self.get_dataset_folder(), f'reports')
            report_creator = ReportCreator(folder, self.debug)
            report_creator.report_references_not_found(not_found_list, label)
            report_creator.test_correctness_of_labeling(not_found_list, self.references_df)
        return df

    def get_bge_criticality_list(self):
        """
        create list of all bger_references
        :return:    return this list
        """
        self.logger.info(f"Processing labeling of bge_criticality")
        bge_ref_list = list(self.references_df['bger_reference'].unique())
        self.logger.info(f"There are {len(bge_ref_list)} references in the bge_criticality_list.")
        return bge_ref_list

    def get_citations_criticality_list(self, df, citations_amounts=[1]):
        """
        create a list of bger_references which were cited a specified amount
        :param df:                dataframe of all bger cases
        :param citations_amounts  array which specifies amount of citations a case must have to be in the list
        :return:                  list of lists of bger_references
        """
        self.logger.info(f"Processing labeling of citation_criticality")

        citations_df = self.process_citations(df)

        # report number of citations
        folder = self.create_dir(self.get_dataset_folder(), f'reports/citations')
        report_creator = ReportCreator(folder, self.debug)
        report_creator.report_citations_count(citations_df, 'all')

        # apply for each row a function which returns True if citation amount is bigger than given number
        critical_lists = []
        for number in citations_amounts:
            critical_df = citations_df[citations_df['counter'] >= number]
            self.logger.info(
                f"There were {len(critical_df.index)} cases found in extracted bger references which were cited at least {number} times.")
            # create list of bge file numbers is unique due to counter
            bge_list = critical_df.loc[:, 'bger_reference'].tolist()
            # make sure list only contains file number once
            bge_list = list(set(bge_list))
            self.logger.info(f"There are {len(bge_list)} references in the bge_criticality_list.")
            critical_lists.append(bge_list)
            report_creator.report_citations_count(citations_df, str(number))
        return critical_lists

    def process_citations(self, df):
        """
        Count for each bge amount of citations in bger
        :param df:      dataframe of all bger cases
        :return:        dataframe containing columns 'bger_reference', 'bge_file_number' and 'count'
        """
        a = len(df.index)
        df.dropna(subset=['citations'], inplace=True)
        self.logger.info(f"There were {a - len(df.index)} cases where no bge citations were found")

        df['ruling_citation'] = df.citations.apply(self.get_citation, type='ruling')
        df.dropna(subset=['ruling_citation'], inplace=True)

        citation_frequencies_df = self.build_tf_matrix(df)

        citation_count_df = self.references_df.copy()

        # iterate over unique values in column 'counter'
        a = citation_frequencies_df['counter'].unique()
        citation_count_df['counter'] = 0

        for i in a:
            # set counter in citation_count_df where citation_frequencies_df has matching reference and count value
            temp_freq_df = citation_frequencies_df[citation_frequencies_df['counter'] == i]
            temp_list = temp_freq_df['bge_file_number'].astype(str).tolist()
            match = citation_count_df.bge_file_number_short.astype(str).isin(temp_list)
            citation_count_df.loc[match, 'counter'] = i

        # apply weight depending on year of cited case
        def apply_weight(row):
            weight = row['year'] - self.first_year
            if weight < 0:
                weight = 0
            return row['counter'] * weight
        citation_count_df['counter'] = citation_count_df.apply(apply_weight, axis=1)

        return citation_count_df

    def build_tf_matrix(self, df):
        if not self.count_all_cits:
            # count citations for one bge only once in bger
            df['ruling_citation'] = df['ruling_citation'].apply(lambda x: set(x))

        # create counter out of list
        df['ruling_citation'] = df['ruling_citation'].apply(lambda x: Counter(x))

        newest_year = df['year'].max()

        # add weight depending on when citation was done
        def weight_citations(row):
            weight = (row['year'] - self.first_year)
            if weight < 0:
                weight = 0
            weighted_counts = row['ruling_citation']
            for k in weighted_counts.keys():
                weighted_counts[k] = weighted_counts[k] * int(weight)
            return weighted_counts
        # df['ruling_citation'] = df.apply(weight_citations, axis=1)

        # create one counter for all citations
        type_corpus_frequency = Counter()
        for index, row in df.iterrows():
            type_corpus_frequency.update(row['ruling_citation'])
        # do this after using Counter() because division will not give int type
        for k in type_corpus_frequency.keys():
            type_corpus_frequency[k] = type_corpus_frequency[k] / (newest_year - self.first_year)

        citation_frequencies_df = pd.DataFrame.from_records(list(dict(type_corpus_frequency).items()),
                                                            columns=['bge_file_number', 'counter'])

        self.logger.info(f"Citation Criticality: There were {len(citation_frequencies_df.index)} unique bge cited")
        return citation_frequencies_df

    def get_law_citation(self, citations_text):
        """
        We don't need law citations in criticality dataset creator
        """
        pass

    def plot_custom(self, report_creator, df, folder):
        """
        Saves statistics and reports about bge_label and citation_label. Specific for criticality_dataset_creator.
        :param report_creator:
        :param folder:          folder to save data
        :param df:              the df containing the dataset
        """
        for attribute in ['legal_area', 'language']:
            report_creator.plot_attribute(df[df['bge_label'] == 'critical'], attribute, name='bge')

        report_creator.plot_label_ordered(df.rename(columns={'bge_label': 'label'}, inplace=False), 'bge_label')
        match = df['citation_label'] == 'non-critical'
        order_dict = dict(label=["citation_0", "citation_1", "citation_10", "citation_100"])
        if self.debug:
            report_creator.plot_label_ordered(df[~match].rename(columns={'citation_label': 'label'}, inplace=False),
                                              'citation_label')
        else:
            report_creator.plot_label_ordered(df[~match].rename(columns={'citation_label': 'label'}, inplace=False),
                                              'citation_label', order=order_dict)


if __name__ == '__main__':
    config = get_config()
    criticality_dataset_creator = CriticalityDatasetCreator(config)
    criticality_dataset_creator.create_dataset(sub_datasets=False, kaggle=False, save_reports=True)
    # criticality_dataset_creator.create_debug_dataset()
