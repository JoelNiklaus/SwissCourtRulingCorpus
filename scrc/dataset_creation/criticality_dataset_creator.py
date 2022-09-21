import ast
import os.path

import pandas as pd
import itertools
import numpy as np
import math
import json


from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.data_classes.ruling_citation import RulingCitation
from root import ROOT_DIR
from pathlib import Path

from scrc.enums.cantons import Canton
from scrc.utils.main_utils import get_config
from scrc.utils.log_utils import get_logger
from collections import Counter

from scrc.utils.sql_select_utils import where_string_spider

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
        self.feature_cols = ['facts']
        self.available_bges = self.load_rulings()
        self.references_df = self.extract_bge_references()
        self.citation_amount = [1000, 100, 10, 1]  # sorted, the highest number first!
        self.counter = 0

    def load_rulings(self):
        """
        Load all bge cases and store in available_bges
        """
        where_string = f"d.decision_id IN {where_string_spider('decision_id', 'CH_BGE')}"
        table_string = 'decision d LEFT JOIN file_number ON file_number.decision_id = d.decision_id'
        decision_df = next(
            self.select(self.get_engine(self.db_scrc), table_string, 'd.*, file_number.text',
                        where_string,
                        chunksize=self.get_chunksize()))
        self.logger.info(f"BGE: There are {len(decision_df.index)} in db (also old or not referenced included).")
        return set(decision_df.text.tolist())

    def extract_bge_references(self):
        """
        Extract data from bge_refereces_found.txt
        :return:        dataframe containing bger_references and bge_file_numbers
        """
        # get dict of bge references with corresponding bge file name
        bge_references_file_path: Path = ROOT_DIR / 'data' / 'datasets' / "bge_references_found.txt"
        if not bge_references_file_path.exists():
            raise Exception("bge references need to be extracted first. Run bge_reference_extractor.")
        df = pd.DataFrame({'bge_file_number_long': [], 'bge_file_number_short': [], 'bger_reference': [], 'year': [], 'bge_chamber': []})
        with bge_references_file_path.open("r") as f:
            for line in f:
                (bge_file_number_long, bger_reference) = line.split()
                year = int(bge_file_number_long.split('-', 5)[1]) + 1874
                bge_chamber = bge_file_number_long.split('_', 5)[2]
                bge_file_number_short = bge_file_number_long.split('_')[3]
                s_row = pd.Series([bge_file_number_long, bge_file_number_short, bger_reference, year, bge_chamber], index=df.columns)
                df = df.append(s_row, ignore_index=True)
        self.logger.info(
            f"References_df: There are {len(df.index)} entries with {len(df.bge_file_number_long.unique())} unique bge_filenumbers and {len(df.bger_reference.unique())} unique bger_references.")
        return df

    def extend_file_number_list(self, file_number_list):
        # this extension is needed because file_numbers have no common syntax.
        bge_list = []
        for i in file_number_list:
            i = i.replace(' ', '_')
            i = i.replace('.', '_')
            (chamber, text) = i.split("_")
            all = [f"{chamber} {text}", f"{chamber}.{text}", f"{chamber}_{text}"]
            bge_list.extend(all)
        return bge_list

    def get_dataset(self, feature_col, save_reports):
        """
        get of all bger cases and set bge_label and citation_label
        :param feature_col:     specify sections to add as column
        :param save_reports:    whether or not to compute and save reports
        :return:                dataframe and list of labels
        """
        df = self.get_df(self.get_engine(self.db_scrc), feature_col)
        a = len(df['file_number'].unique())
        self.logger.info(f"BGer: There are {len(df.index)} bger cases with {a} unique file numbers.")

        # bge criticality
        bge_list = self.get_bge_criticality_list()
        df = self.set_criticality_label(df, bge_list, 'bge_label')

        # citation criticality
        criticality_lists = self.get_citations_criticality_list(df, self.citation_amount)
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
                critical_df['citation_label'] = critical_df['citation_label'].map({'critical': f"critical-{self.citation_amount[i]}"}, na_action=None)
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
        return df, label_list

    def filter_columns(self, df):
        """
        Filters given dataframe, to get rid of not needed data
        :param df:      dataframe of all bger cases
        :return:        filtered dataframe
        """
        columns = list(df.columns)
        # huggingface: year, legal-area, origin-region, origin-canton, bge-label, citation-label, lang, cons, facts
        # plots: legal_area, origin_region, origin_canton, origin_court, origin_chamber
        needed_cols = ['year', 'legal_area', 'origin_region', 'origin_canton', 'origin_court', 'origin_chamber', 'lang',
                       'bge_label', 'citation_label']
        for feature_col in self.feature_cols:
            needed_cols.append(f"{feature_col}")
            needed_cols.append(f"{feature_col}_num_tokens_spacy")
            needed_cols.append(f"{feature_col}_num_tokens_bert")
        for item in needed_cols:
            columns.remove(item)
        df.drop(columns, axis=1, inplace=True)
        return df

    def clean_criticality_df(self, df):
        # clean df only after processing all citations (from all cases)
        # TODO create real court and chamber from float input
        df['origin_court'] = df.origin_court.astype(str)
        df['origin_chamber'] = df.origin_chamber.astype(str)

        def get_canton_value(x):
            if not math.isnan(float(x)):
                canton = Canton(int(x))
                return canton.name
            else:
                return np.nan

        df.origin_canton = df.origin_canton.apply(get_canton_value)

        for feature_col in self.feature_cols:
            df[f'{feature_col}_length'] = df[feature_col].astype(str).apply(len)
            match = df[f'{feature_col}_length'] > self.minFeatureColLength
            df.loc[~match, feature_col] = np.nan

        a = len(df.index)
        df = df.dropna(subset=self.feature_cols, how='all')
        self.logger.info(f"There were {a-len(df.index)} cases dropped because all feature cols were too short.")
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
        :param df:          dataframe of all bger cases
        :citations_amounts  array which specifies amount of citations a case must have to be in the list
        :return:            list of lists of bger_references
        """
        self.logger.info(f"Processing labeling of citation_criticality")

        citations_df = self.process_citations(df)

        # report number of citations
        folder: Path = ROOT_DIR / 'data' / 'datasets' / 'criticality_prediction' / "-".join(self.feature_cols)
        split_folder = self.create_dir(folder, f'reports/citations_amount')
        citations_df = citations_df.dropna(subset=['counter'])
        temp_df = citations_df.copy()
        temp_df.loc[:, 'counter'] = temp_df.counter.astype(str)
        self.plot_barplot_attribute(temp_df, split_folder, 'counter')

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
        Count for bge amount of citations in other bger
        :param df:      dataframe of all bger cases
        :return:        dataframe containing columns 'bger_reference', 'bge_file_number' and 'count'
        """
        a = len(df.index)
        df = df.dropna(subset=['citations'])
        df.loc[:, 'ruling_citation'] = df.citations.apply(self.get_citations)
        # get rid of reset the index so that we don't get out of bounds errors
        df = df.dropna(subset=['ruling_citation']).reset_index(drop=True)
        self.logger.info(f"There were {a - len(df.index)} cases where no bge citations were found")
        self.logger.info(f"Building the term-frequency matrix.")
        df['counter'] = df['ruling_citation'].apply(lambda x: dict(Counter(x)))

        # counts how often a ruling is cited
        # assert no entry with 0 exists
        type_corpus_frequencies = dict(Counter(itertools.chain(*df['ruling_citation'].tolist())))
        data = {
            "bge_file_number": list(type_corpus_frequencies.keys()),
            "counter": list(type_corpus_frequencies.values())
        }

        citation_frequencies_df = pd.DataFrame.from_dict(data)
        self.logger.info(f"Citation Criticality: There were {len(citation_frequencies_df.index)} unique bge cited")

        citation_count_df = self.references_df
        # iterate over unique count values
        a = citation_frequencies_df['counter'].unique()
        for i in a:
            # set counter in citation_count_df where citation_frequencies_df has matching reference and count value
            temp_freq_df = citation_frequencies_df[citation_frequencies_df['counter'] == i]
            temp_list = temp_freq_df['bge_file_number'].astype(str).tolist()
            file_number_match = citation_count_df.bge_file_number_short.astype(str).isin(temp_list)
            citation_count_df.loc[file_number_match, 'counter'] = int(i)
        return citation_count_df

    def get_citations(self, citations_as_string):
        """
        extract for each bger all ruling citations
        :param citations_as_string:         citations how they were found in text of bger
        :return:                            dataframe with additional column 'ruling_citation'
        """
        self.counter = self.counter + 1
        if int(self.counter) % 10000 == 0:
            print("Processed another 10000")
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
                    self.logger.info(f"Citation has invalid syntax: {citation}")
                    continue
        except ValueError as ve:
            self.logger.info(f"Citations could not be extracted to dict: {citations_as_string}")
        if cits:  # only return something if we actually have citations
            return cits

    def get_file_number(self, citation):
        """
        find for each citation string the matching citation from the start of bge (first page)
        :param citation:         citation as string as found in text
        :return:                 RulingCitation always in German
        """
        # TODO scrape for all bge file number
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
            # make sure new page number is not unrealistic far away.
            if page_number - new_page_number < 20:
                return RulingCitation(f"{year} {volume} {new_page_number}", 'de')
            return found_citation

    def save_huggingface_dataset(self, splits, feature_col_folder):
        """
        save data as huggingface dataset with columns: 'id', 'year': year, 'language',
        region', 'canton', 'legal area', 'bge_label', 'considerations' and 'facts'
        ATTENTION: only works for feature_cols = [considerations, facts]
        :param splits:                  specifying splits of dataset
        :param feature_col_folder:      name of folder
        """
        huggingface_dir = self.create_dir(feature_col_folder, 'huggingface')
        for split in ['train', 'val', 'test']:
            records = []
            df = splits[split]
            i = 0
            # ATTENTION this works only for 1 or 2 entries in feature_col
            if len(self.feature_cols) > 1:
                i = 1
            tuple_iterator = zip(df.index, df['year'], df['legal_area'], df['origin_region'], df['citation_label'],
                                 df['origin_canton'], df['bge_label'], df['lang'], df[self.feature_cols[0]],
                                 df[self.feature_cols[i]])

            for case_id, year, legal_area, region, citation_label, canton, bge_label, lang, a, b in tuple_iterator:
                if not isinstance(canton, str) and (canton is None or math.isnan(canton)):
                    canton = 'n/a'
                if not isinstance(region, str) and (region is None or math.isnan(region)):
                    region = 'n/a'
                if not isinstance(legal_area, str) and (legal_area is None or math.isnan(legal_area)):
                    legal_area = 'n/a'
                record = {
                    'id': case_id,
                    'year': year,
                    'language': lang,
                    'region': region,
                    'canton': canton,
                    'legal area': legal_area,
                    'bge_label': bge_label,
                    'citation_label': citation_label,
                }
                record[self.feature_cols[0]] = a
                if i == 1:
                    record[self.feature_cols[1]] = b

                records.append(record)
            with open(f'{huggingface_dir}/{split}.jsonl', 'w') as out_file:
                for record in records:
                    out_file.write(json.dumps(record) + '\n')

    def plot_custom(self, df, split_folder, folder):
        """
        Saves statistics about the dataset in the form of csv tables and png graphs.
        :param folder:  the base folder to save the report to
        :param split_folder:   the name of the split
        :param df:              the df containing the dataset
        :return:
        """

        barplot_attributes = ['legal_area', 'origin_region', 'origin_canton', 'origin_court', 'origin_chamber']

        for attribute in barplot_attributes:
            self.plot_barplot_attribute(df[df['bge_label'] == 'critical'], split_folder, attribute, 'bge')
            for i in self.citation_amount:
                self.plot_barplot_attribute(df[df['citation_label'] == f"critical-{i}"], split_folder, attribute,
                                            f"citation_{i}")

        for feature_col in self.feature_cols:
            dict = {f'{feature_col}_num_tokens_bert': 'num_tokens_bert',
                    f'{feature_col}_num_tokens_spacy': 'num_tokens_spacy'}
            self.plot_input_length(df.rename(columns=dict), split_folder, feature_col=feature_col)

        # plot labels, need to rename column to use function
        self.plot_labels(df.rename(columns={'bge_label': 'label'}, inplace=False), split_folder, label_name='bge_label')
        self.plot_labels(df.rename(columns={'citation_label': 'label'}, inplace=False), split_folder, label_name='citation_label')

        # report references_df distribution, needs to be done only after running reference-extractor.
        if not os.path.exists(folder / 'reports' / f'bge_references'):
            folder_tmp = self.create_dir(folder, f'reports/bge_references')
            self.references_df['bge_chamber'] = self.references_df.bge_chamber.astype(str)
            self.references_df['year'] = self.references_df.year.astype(str)
            barplot_attributes = ['bge_chamber', 'year']
            for attribute in barplot_attributes:
                self.plot_barplot_attribute(self.references_df, folder_tmp, attribute)
        else:
            self.logger.info("There exists already reports for bge_references. To create new you have to delete old.")

    def report_citations_count(self, df, counts):
        # report distribution of citation, here because it's deleted from df later.
        folder: Path = ROOT_DIR / 'data' / 'datasets' / 'criticality_prediction' / "-".join(self.feature_cols)
        split_folder = self.create_dir(folder, f'reports/citation_{counts}')
        df['year'] = df.year.astype(str)
        self.plot_barplot_attribute(df, split_folder, 'year')

    def print_not_found_list(self, not_found_list, label):
        file_path = ROOT_DIR / 'data' / 'datasets' / 'criticality_prediction' / "-".join(self.feature_cols) / 'reports' / "not_found_references.txt"
        if not file_path.exists():
            path = ROOT_DIR / 'data' / 'datasets' / 'criticality_prediction' / "-".join(self.feature_cols)
            self.create_dir(path, 'reports')
        with open(file_path, 'a') as f:
            f.write(f"List of not found references for {label}:\n")
            for item in not_found_list:
                f.write(f"{item}\n")

    def test_correctness_of_labeling(self, not_found_list):
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
    criticality_dataset_creator.create_dataset(sub_datasets=False, kaggle=False, huggingface=True, save_reports=True)
    # criticality_dataset_creator.create_debug_dataset()
