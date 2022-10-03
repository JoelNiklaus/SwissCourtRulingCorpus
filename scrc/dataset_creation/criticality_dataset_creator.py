import os.path

import pandas as pd
import numpy as np
import datasets

from scrc.dataset_creation.dataset_creator import DatasetCreator
from pathlib import Path

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

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.debug = True
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
        return df

    def extend_file_number_list(self, file_number_list):
        """
        extend a given list of bger_file_numbers with unknown syntax to a list which contains all possible syntax
        :param file_number_list:    list of bger_file_numbers with unknown syntax
        :return:                    list with all 3 syntaxes of each bger_file_number found in given list
        """
        # this extension is needed because file_numbers have no common syntax.
        bge_list = []
        for i in file_number_list:
            i = i.replace(' ', '_')
            i = i.replace('.', '_')
            (chamber, text) = i.split("_")
            all = [f"{chamber} {text}", f"{chamber}.{text}", f"{chamber}_{text}"]
            bge_list.extend(all)
        return bge_list

    def prepare_dataset(self, save_reports):
        """
        get of all bger cases and set bge_label and citation_label
        :param save_reports:    whether or not to compute and save reports
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

        df = self.clean_criticality_df(df)
        bge_labels, _ = list(np.unique(np.hstack(df.bge_label), return_index=True))
        citation_labels, _ = list(np.unique(np.hstack(df.citation_label), return_index=True))
        label_list = [bge_labels, citation_labels]
        return datasets.Dataset.from_pandas(df), label_list

    def filter_columns(self, df):
        """
        Filters cols of dataframe, to get rid of not needed columns
        :param df:      dataframe of all bger cases
        :return:        filtered dataframe
        """
        columns = list(df.columns)
        needed_cols = ['year', 'legal_area', 'origin_region', 'origin_canton', 'origin_court', 'origin_chamber', 'lang',
                       'bge_label', 'citation_label']
        for feature_col in self.get_feature_col_names():
            needed_cols.append(f"{feature_col}")
            needed_cols.append(f"{feature_col}_num_tokens_spacy")
            needed_cols.append(f"{feature_col}_num_tokens_bert")
        for item in needed_cols:
            columns.remove(item)
        df.drop(columns, axis=1, inplace=True)
        return df

    def clean_criticality_df(self, df):
        """
        clean df (after labels are set) to get rid of odd values and define minimum length for feature_cols
        ATTENTION: This should be done only after processing all citations. Because otherwise some citations in cases we
        removed would not be considered.
        :param df:  dataframe of all cases
        :return:    cleaned dataframe
        """
        # Get rid of cases which can't be used for training because all feature cols are too short
        # Those cases are needed to create labels, that's why we cannot delete them earlier
        for feature_col in self.get_feature_col_names():
            df[f'{feature_col}_length'] = df[feature_col].astype(str).apply(len)
            match = df[f'{feature_col}_length'] > self.minFeatureColLength
            df.loc[~match, feature_col] = np.nan

        a = len(df.index)
        df = df.dropna(subset=self.feature_cols, how='all')
        self.logger.info(f"There were {a - len(df.index)} cases dropped because all feature cols were too short.")
        df = df.reset_index(drop=True)  # reindex to get nice indices
        df = self.filter_columns(df)
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
        critical_df = df[df[label] == 'critical']
        self.logger.info(
            f"{label}: There are {len(critical_df.index)} critical cases with {len(critical_df.file_number.unique())} unique file_numbers.")
        if label == "bge_label":
            df.loc[~file_number_match, label] = 'non-critical'

            # Check which file numbers could not been found for bge criticality
            labeled_critical_list = self.extend_file_number_list(list(critical_df.file_number.unique()))
            not_found_list = [item for item in criticality_list if item not in labeled_critical_list]
            self.print_not_found_list(not_found_list, label)
            self.test_correctness_of_labeling(not_found_list)
        return df

    def get_bge_criticality_list(self):
        """
        create list of all bger_references
        :return:    return this list
        """
        self.logger.info(f"Processing labeling of bge_criticality")
        bge_ref_list = list(self.references_df['bger_reference'].unique())
        self.logger.info(f"There are {len(bge_ref_list)} references in the bge_criticality_list.")
        # Extend list because it is not sure with which syntax filenumber is saved in db
        bge_list = self.extend_file_number_list(bge_ref_list)
        return bge_list

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
        split_folder = self.create_dir(self.get_dataset_folder(), f'reports/citations_amount')
        self.plot_barplot_attribute(citations_df, split_folder, 'counter')

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
            bge_list = self.extend_file_number_list(bge_list)
            self.logger.info(f"There are {len(bge_list)} references in the bge_criticality_list.")
            critical_lists.append(bge_list)
            self.report_citations_count(critical_df, str(number))
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
        return citation_count_df

    def build_tf_matrix(self, df):
        if not self.count_all_cits:
            # count citations for one bge only once in bger
            df['ruling_citation'] = df['ruling_citation'].apply(lambda x: set(x))

        # create counter out of list
        df['ruling_citation'] = df['ruling_citation'].apply(lambda x: Counter(x))

        newest_year = df['year'].max()

        # multiply weight to value in counter
        def weight_citations(row):
            weight = (row['year'] - self.first_year)
            if weight < 0:
                weight = 0
            weighted_counts = row['ruling_citation']
            for k in weighted_counts.keys():
                weighted_counts[k] = weighted_counts[k] * int(weight)
            return weighted_counts

        df['ruling_citation'] = df.apply(weight_citations, axis=1)
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

    def plot_custom(self, df, split_folder, folder):
        """
        Saves statistics and reports about bge_label and citation_label. Specific for criticality_dataset_creator.
        :param folder:          the base folder to save the report to
        :param split_folder:    the name of the split
        :param df:              the df containing the dataset
        """

        barplot_attributes = ['legal_area', 'origin_region', 'origin_canton', 'origin_court', 'origin_chamber']

        for attribute in barplot_attributes:
            self.plot_barplot_attribute(df[df['bge_label'] == 'critical'], split_folder, attribute, 'bge')
            for i in self.citation_amount:
                self.plot_barplot_attribute(df[df['citation_label'] == f"critical-{i}"], split_folder, attribute,
                                            f"citation_{i}")

        for feature_col in self.get_feature_col_names():
            dict = {f'{feature_col}_num_tokens_bert': 'num_tokens_bert',
                    f'{feature_col}_num_tokens_spacy': 'num_tokens_spacy'}
            self.plot_input_length(df.rename(columns=dict), split_folder, feature_col=feature_col)

        # plot labels, need to rename column to use function
        self.plot_labels(df.rename(columns={'bge_label': 'label'}, inplace=False), split_folder, label_name='bge_label')
        self.plot_labels(df.rename(columns={'citation_label': 'label'}, inplace=False), split_folder,
                         label_name='citation_label')

        # report references_df distribution, needs to be done only after running reference-extractor.
        if not os.path.exists(folder / 'reports' / f'bge_references'):
            folder_tmp = self.create_dir(folder, f'reports/bge_references')
            barplot_attributes = ['bge_chamber', 'year']
            for attribute in barplot_attributes:
                self.plot_barplot_attribute(self.references_df, folder_tmp, attribute)
        else:
            self.logger.info("There exists already reports for bge_references. To create new you have to delete old.")

    def report_citations_count(self, df, counts):
        """
        Saves reports about each citation_label class.
        This needs to be done plot_custom because count to cover all found citations independent whether bge is in db or not.
        :param df:      dataframe containing all the data
        :param counts:   specifying which citation_label class to use
        """
        # report distribution of citation, here because it's deleted from df later.
        split_folder = self.create_dir(self.get_dataset_folder(), f'reports/citation_{counts}')
        self.plot_barplot_attribute(df, split_folder, 'year')

    def print_not_found_list(self, not_found_list, label):
        """
        Save list of all bger_references which were extracted but could not be found as bge in db
        :param not_found_list:  list of references
        :param label:           specifying for with label class the given list was created for
        """
        file_path = self.get_dataset_folder() / 'reports' / "not_found_references.txt"
        if not file_path.exists():
            self.create_dir(self.get_dataset_folder(), 'reports')
        with open(file_path, 'a') as f:
            f.write(f"List of not found references for {label}:\n")
            for item in not_found_list:
                f.write(f"{item}\n")

    def test_correctness_of_labeling(self, not_found_list):
        """
        in order to assure correctness, test which cases could not be found and why
        :param not_found_list:      list of references were no bger case could be found for.
        """
        self.logger.info(f"Not Found List: There are {len(not_found_list)} not found cases.")
        # get list where bge file number is multiple times in references df
        file_number_list = self.references_df.bge_file_number_long.tolist()
        file_number_counter = Counter(file_number_list)
        double_bge_file_numbers = [k for k, v in file_number_counter.items() if v > 1]
        # get list where bger references is multiple time in references df
        reference_list = self.references_df.bger_reference.tolist()
        reference_counter = Counter(reference_list)
        double_bger_references = [k for k, v in reference_counter.items() if v > 1]
        # find cases where bge_reference and file number did occur mutliple times
        file_number_match = self.references_df.bge_file_number_long.astype(str).isin(list(double_bge_file_numbers))
        doubled_df = self.references_df[file_number_match]
        file_number_match = doubled_df.bger_reference.astype(str).isin(list(double_bger_references))
        doubled_df = doubled_df[file_number_match]
        self.logger.info(f"There are {len(doubled_df.index)} double entries from reference extractor. ")
        # make sure an entry in not_found_list is not wrongly found as entry in references
        for item in not_found_list:
            assert item not in file_number_list
        # check how many cases of the not found list is from before 2000
        file_number_match = self.references_df.bger_reference.astype(str).isin(not_found_list)
        not_found_df = self.references_df[file_number_match]
        new = not_found_df.loc[not_found_df['year'] > 2000]
        self.logger.info(f"There are {len(new.index)} cases from before 2000 which where not found.")


if __name__ == '__main__':
    config = get_config()
    criticality_dataset_creator = CriticalityDatasetCreator(config)
    criticality_dataset_creator.create_dataset(sub_datasets=False, kaggle=False, save_reports=True)
    # criticality_dataset_creator.create_debug_dataset()
