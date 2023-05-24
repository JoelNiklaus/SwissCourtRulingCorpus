import abc
import copy
import math
import os
import sys
import gc
import time
from datetime import date
from pathlib import Path
from typing import Union
import ast
from tqdm import tqdm
import csv
import seaborn as sns
import datasets
import re
from datasets import concatenate_datasets
from root import ROOT_DIR
from multiprocessing import Pool
from scrc.dataset_creation.report_creator import ReportCreator
from scrc.data_classes.ruling_citation import RulingCitation
import numpy as np
import pandas as pd
from scrc.enums.section import Section
from scrc.preprocessors.abstract_preprocessor import AbstractPreprocessor
from scrc.utils.log_utils import get_logger
import json
from scrc.utils.main_utils import retrieve_from_cache_if_exists, save_df_to_cache, get_canton_from_chamber, \
    get_court_from_chamber, print_memory_usage
from scrc.utils.sql_select_utils import get_legal_area, join_tables_on_decision, legal_areas, get_region, \
    where_string_spider, where_string_court
from scrc.utils.court_names import court_names_backup, get_error_courts, get_empty_courts
from scrc.enums.split import Split
from datasets import load_dataset


import scrc.utils.monkey_patch  # IMPORTANT: DO NOT REMOVE: prevents memory leak with pandas

csv.field_size_limit(sys.maxsize)

# pd.options.mode.chained_assignment = None  # default='warn'
sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
sns.set_style("whitegrid")
"""
Extend datasets with big cantonal courts? => only if it does not take too much time (1-2 days per court)
Datasets to be created:
- Judgments
    - Judgment prediction BGer:
        - text classification
        - input (considerations/facts) to label (judgment)
        - Why? 
- Citations
    - Citation prediction
        - multilabel text classification
        - input (entire text with citations removed) to label (cited laws and court decisions)
            labels:
                - all possible rulings
                - most frequent rulings
                - all law articles
                - most frequent law articles
                - law books (without article number)
        - Features:
            - Zero/One/Few Shot
        - Why?: 
    - Citation labeling (similar: https://github.com/reglab/casehold):
        - fill mask
        - input (entire text with citations removed by mask token)
        - Tasks: 
            - predict citation type (law or ruling) or citation
            - multiple choice question answering (choose from say 5 options)
        - Why?: difficult LM Pretraining task
    - Semantic Textual Similarity
        - regression (similarity of two decisions)
        - input: two decision texts
        - label: similarity (between 0 and 1)
        - the more citations are in common, the more similar two decisions are
        - the more similar the common citations the higher the similarity of the decisions
- Criticality Prediction
    - level of appeal prediction
        - text classification/regression (error of model makes more sense))
        - input: facts and considerations (remove information which makes it obvious)
        - label: predict level of appeal of current case
        - Why?: 
    - controversial case prediction
        - (binary) text classification
        - input: entire case of a local court (1st level of appeal)
        - label: predict top level of appeal the case will end up in
        - Why?: very interesting conclusions contentwise 
            (e.g. if we predict that the case might end up in the supreme court => speed up process because the case is very important)
    - case importance prediction (criticality)
        - regression task
        - input: entire text
        - label: level of criticality (normalized citation count (e.g. 1 to 10, 0 to 1))
        - Why?: Can the criticality be determined only by the text? Are specific topics very controversial?
- Chamber/court prediction (proxy for legal area)
    - text classification
    - input (facts) to label (chamber/court code)
    - Features:
        - Zero/One/Few Shot
    - Why?: proxy for legal area prediction to help lawyers find suitable laws
- LM Pretraining:
    - masked language modeling
    - input (entire text)
    - Features:
        - largest openly published corpus of court decisions
    - Why?: train Swiss Legal BERT


- To maybe be considered later 
    - Section splitting: 
        - comment: if it can be split with regexes => not interesting
            - if the splitting is non-standard => more interesting, but manual annotation needed!
            - similar to paper: Structural Text Segmentation of Legal Documents
            - structure can be taken from html (=> labels) and input can be raw text => easy way to get a lot of ground truth!
                - for example splitting into coherent paragraphs
        - token classification (text zoning task), text segmentation
        - input (entire text) to label (section tags per token)
        - Why?: 
    - Date prediction
        - text classification
        - input (entire text) to label (date in different granularities: year, quarter, regression)
        - Features:
            - Zero/One/Few Shot
        - Why?: learn temporal data shift
        - not sure if interesting

Features:
- Time-stratified: train, val and test from different time ranges to simulate more realistic test scenario
- multilingual: 3 languages
- diachronic: more than 20 years
- very diverse: 3 languages, 26 cantons, 112 courts, 287 chambers, most if not all legal areas and all Swiss levels of appeal
"""

"""
Further projects: (inspired by https://arxiv.org/pdf/2106.10776.pdf)
Investigate Legal Citation Prediction as a Natural Language Generation Problem
Citation Prediction only on context of the citation not on the entire document (simulating the writing of a decision by a clerk)
"""


class DatasetCreator(AbstractPreprocessor):
    """
    TODO look at this project for easy data reports: https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/pages/introduction.html
    TODO alternative for project above: https://dataprep.ai/
    Retrieves the data and preprocesses it for subdatasets of SCRC.
    Also creates the necessary files for a kaggle dataset and a huggingface dataset.
    """

    def __init__(self, config: dict, debug: bool = True):
        __metaclass__ = abc.ABCMeta
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.debug = debug
        self.seed = 42
        self.debug_chunksize = 100
        self.real_chunksize = 1_000_000
        self.counter = 0
        self.start_years = {Split.TRAIN.value: 1900, Split.VALIDATION.value: 2016, Split.TEST.value: 2018, Split.SECRET_TEST.value: 2023}
        self.current_year = date.today().year
        self.metadata = ['year', 'chamber', 'court', 'canton', 'region',
                         'origin_chamber', 'origin_court', 'origin_canton',
                         'law_area', 'law_sub_area']
        self.num_cores = 16
        self.delete_row_only_if_all_feature_cols_below_cutoff = False

        def build_info_df(table_name, col_name):
            info_df = next(self.select(self.get_engine(self.db_scrc), table_name))
            info_dict = {}
            for index, row in info_df.iterrows():
                info_dict[int(row[f'{table_name}_id'])] = str(row[col_name])
            return info_dict

        self.chamber_dict = build_info_df('chamber', 'chamber_string')
        # self.court_dict = build_info_df('court', 'court_string')
        # self.canton_dict = build_info_df('canton', 'short_code')

        self.overwrite_cache = True  # to be overridden
        self.split_type = None  # to be overridden
        self.dataset_name = None  # to be overridden
        self.feature_cols = [Section.FULL_TEXT]  # to be overridden
        self.filter_cols = []  # to be overridden
        self.labels = []  # to be overridden
        self.available_rulings_dict = {}  # to be overridden
        self.reports_folder: Path = Path()

    @abc.abstractmethod
    def prepare_dataset(self, save_reports, court_string):
        pass

    def get_chunksize(self):
        if self.debug:
            return int(self.debug_chunksize)  # run on smaller dataset for testing
        else:
            return int(self.real_chunksize)

    def load_rulings(self):
        """
        Load all bge cases and store in available_bges
        """
        ruling_dataset = load_dataset('rcds/swiss_leading_decisions', split='train')
        decision_df = pd.DataFrame(ruling_dataset)
        self.logger.info(f"BGE: There are {len(decision_df.index)} in db (also old or not referenced included).")

        available_rulings_dict = {}

        def filter_rulings(row):
            file_number = str(row['file_number'])
            parts = file_number.split("_")
            if len(parts) == 4:
                year = int(parts[1])
                volume = parts[2]
                page = int(parts[3])
                if year in available_rulings_dict:
                    if volume in available_rulings_dict[year]:
                        available_rulings_dict[year][volume].append(page)
                    else:
                        available_rulings_dict[year][volume] = [page]
                else:
                    available_rulings_dict[year] = {volume: [page]}
                return True
            return False
        useful_cases = decision_df.apply(filter_rulings, axis='columns')
        decision_df = decision_df[useful_cases]

        return decision_df, available_rulings_dict

    def get_citation(self, citations, type):
        """
        extract for each bger all ruling citations
        :param citations_as_string:         citations how they were found in text of bger
        :param cit_type:
        :return:                            dataframe with additional column 'ruling_citation'
        """
        cits = []
        try:
            # citations = ast.literal_eval(citations_as_string)  # parse dict string to dict again
            for citation in citations:
                try:
                    cit = citation['text']
                    citation_type = citation['name']
                    cit = ' '.join(cit.split())  # remove multiple whitespaces inside
                    if citation_type == "ruling" and type == 'ruling':
                        cited_file = self.get_file_number(cit)
                        cits.append(cited_file)
                    elif citation_type == "law" and type == 'law':
                        try:
                            tmp = self.get_law_citation(cit)
                            if tmp is not None:
                                cits.append(tmp)
                        except ValueError as ve:
                            self.logger.info(f"Citation could not be found in laws: {citation}")
                except ValueError as ve:
                    self.logger.info(f"Warning: Citation has invalid syntax: {citation}")
                    continue
        except ValueError as ve:
            self.logger.info(f"Citations could not be extracted to dict: {citations}")
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
        year = int(found_citation.year)
        volume = str(found_citation.volume)  # problem if we get volume like IA because it could be stored as Ia
        page_number = int(found_citation.page_number)
        if year in list(self.available_rulings_dict.keys()):
            if volume in list(self.available_rulings_dict[year].keys()):
                if page_number in list(self.available_rulings_dict[year][volume]):
                    return found_citation.cit_string()
                else:
                    new_page_number = -1
                    for page in self.available_rulings_dict[year][volume]:
                        if new_page_number < page <= page_number:
                            new_page_number = page
                    if (page_number - new_page_number) < 20 and new_page_number > 0:
                        result = RulingCitation(f"{year} {volume} {new_page_number}", 'de')
                        self.counter = self.counter + 1
                        return result.cit_string()

    def get_law_citation(self, citations_text):
        """
        handle single law citation
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def get_dataset_folder(self):
        if self.debug:
            # make sure that we don't overwrite progress in the real directory
            return self.create_dir(self.tmp_subdir, self.dataset_name)
        return self.create_dir(self.datasets_subdir, self.dataset_name)

    def create_dataset(self, court_list=None, concatenate=False, sub_datasets=False, kaggle=False, save_reports=False):
        """
        Retrieves the respective function named by the dataset and executes it to get the df for that dataset.
        :return:
        """
        if court_list is None:
            court_list = ["CH_BGer"]  # default to BGer

        self.logger.info(f"Creating {self.dataset_name} dataset")

        # TODO in the future: maybe save text as list of paragraphs
        # TODO make sure that the same data is saved to kaggle, csv and huggingface format!

        datasets_list = []
        if self.dataset_name == "doc2doc_ir" or self.dataset_name == "criticality_prediction" or self.dataset_name == "citation_extraction":
            dataset, labels, state_tuples = self.create_single_dataset(court_list[0], concatenate, datasets_list, kaggle, save_reports, sub_datasets)
        else:
            with Pool(processes=self.num_cores) as pool:
                results = [pool.apply_async(self.create_single_dataset, (court_string, concatenate, datasets_list, kaggle, save_reports,
                                                                     sub_datasets)) for court_string in court_list]
                datasets = [result.get()[0] for result in results if result.get()[0] is not None]
                labels = [result.get()[1] for result in results if result.get()[1] is not None][0]
                state_tuples = [result.get()[2] for result in results if result.get()] # tuple, e.g. (court_name, state)
            pool.close()

        # group state tuples by state and log them
        state_dict = {}
        for state_tuple in state_tuples:
            if state_tuple[1] not in state_dict:
                state_dict[state_tuple[1]] = []
            state_dict[state_tuple[1]].append(state_tuple[0])
        for state in state_dict:
            self.logger.info(f"{state}: {state_dict[state]}")

        if concatenate:
            self.logger.info("Concatenating datasets")
            dataset = concatenate_datasets(datasets)

            # TODO: investigate if this is really necessary
            print_memory_usage([dataset, datasets_list])
            del datasets_list
            gc.collect()

            # check if export folder already exists and increment the name index if it does
            version = 1
            while os.path.exists(f"{self.get_dataset_folder()}/v{version}"):
                version += 1
            export_path = Path(f"{self.get_dataset_folder()}/v{version}")

            self.save_dataset(dataset, labels, export_path, self.split_type, kaggle=kaggle, save_reports=save_reports)

        else:
            assert len(court_list) == 1 and (self.dataset_name == "doc2doc_ir" or self.dataset_name == "criticality_prediction" or self.dataset_name == "citation_extraction")
            court_string = court_list[0]
            save_path = self.create_dir(self.get_dataset_folder(), court_string)
            self.save_dataset(dataset, labels, save_path, self.split_type,
                              sub_datasets=sub_datasets, kaggle=kaggle, save_reports=save_reports)

    def create_single_dataset(self, court_string, concatenate, datasets_list, kaggle, save_reports, sub_datasets):
        """
        Creates a dataset for a single court.
        :return: dataset, labels
        """
        try:
            self.logger.info(f"Creating dataset for {court_string}")
            dataset, labels = self.prepare_dataset(save_reports, court_string=court_string)

            if len(dataset) <= 1:
                self.logger.info(f"Dataset for {court_string} could not be created")
                return None, None, (court_string, "empty")
            else:
                if concatenate:  # save all of them together in the end
                    datasets_list.append(dataset)
                else:  # save each dataset separately
                    self.logger.info(f"Dataset for {court_string} created")
            return dataset, labels, (court_string, "created")
        except Exception as e:
            self.logger.error(f"Exception for {court_string}: {e}")
            raise e
            return None, None, (court_string, "exception")

    def get_all_courts(self):
        try:
            engine = self.get_engine(self.db_scrc)
            court_names = next(self.select(engine, "court", "court_string", None))["court_string"].tolist()
        except StopIteration:
            self.logger.info("No court names found; using default list.")
            court_names = court_names_backup
        return court_names

    def get_court_list(self, concatenate):
        """
        get_court_list returns all courts that can be generated without any problems based on the current state of knowledge
        :return: list of str objects of court names. e.g. ["CH_BGer", "BL_OG"]
        """

        court_list_tmp = self.get_all_courts()  # get names of all courts

        if not concatenate:
            # taking all folder names from /data/datasets as a list to know which courts are already generated
            courts_done = os.listdir(str(self.datasets_subdir / self.dataset_name))
            self.logger.info(f"Already generated courts: {courts_done}")
        else:
            courts_done = []

        # remove courts from list that are already generated
        court_list = []
        for court in court_list_tmp:
            if court not in courts_done:
                court_list.append(court)

        return court_list

    def create_multiple_datasets(self, court_list=None, concatenate=False, overview=True, save_reports=True,
                                 sub_datasets=False):
        """
        :param court_list:    default: every court without any problems, or to specify court_strings in a list e.g. ["TI_TE", "LU_JSD"]
        :param concatenate:   if True, all courts datasets are concatenated into one file
        :param overview:      if True, creates overview of all generated datasets and exports them in a csv file
        """
        def sort_by_size(court_list):
            """
            Sets the biggest courts first to save time when multiprocessing
            :param court_list: list of court names
            :return: sorted list of court names
            """
            biggest = ["CH_BGer", "VD_TC", "CH_BVGE", "GE_CJ", "ZH_SVG", "TI_TRAC", "TI_TCAS", "BE_VG", "FR_TC",
                       "ZH_VG", "SG_KGN", "SG_VSG", "CH_BSTG", "TI_TCA"]
            sorted_list = []
            for court in biggest:
                if court in court_list:
                    sorted_list.append(court)
            for court in court_list:
                if court not in biggest:
                    sorted_list.append(court)
            return sorted_list

        start_time = time.time()
        if court_list is None:
            court_list = self.get_court_list(concatenate=concatenate)
        court_list = sort_by_size(court_list)
        self.create_dataset(court_list, concatenate=concatenate, sub_datasets=sub_datasets, save_reports=save_reports)
        if overview:
            self.create_overview()
        self.logger.info(f"Time elapsed: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")

    def save_huggingface_dataset(self, splits, folder):
        """
        save data as huggingface dataset with columns:
        'id', 'date', 'year', 'language',
         'origin_canton', 'origin_court', 'origin_chamber', 'law_area',
        'bge_label', 'citation_label', all feature cols
        :param splits:      specifying splits of dataset
        :param folder:      name of folder
        """
        huggingface_dir = self.create_dir(folder, 'huggingface')
        self.logger.info(f"Generating huggingface dataset at {huggingface_dir}")

        def parallelize_saving(splits, func, split_value, n_cores=self.num_cores):
            pool = Pool(n_cores)
            pool.apply_async(func, (splits[split_value], split_value, huggingface_dir, folder))
            pool.close()
            pool.join()

        parallelize_saving(splits, self.save_huggingface_split, Split.TEST.value)
        parallelize_saving(splits, self.save_huggingface_split, Split.TRAIN.value)
        parallelize_saving(splits, self.save_huggingface_split, Split.VALIDATION.value)


    def save_huggingface_split(self, dataset, split: str, huggingface_dir, folder):
        cols_to_include = ['decision_id', 'language'] + self.metadata + self.labels + self.get_feature_col_names()
        if not self.dataset_name == "doc2doc_ir" or not self.dataset_name == "citation_extraction":
            for feature_col in self.feature_cols:
                cols_to_include = cols_to_include + [f'origin_{feature_col}']
        cols_to_remove = [col for col in dataset.column_names if col not in cols_to_include]
        dataset = dataset.remove_columns(cols_to_remove)
        hf_file = f'{huggingface_dir}/{split}.jsonl'
        self.logger.info(f"Saving {split} dataset at {hf_file}")
        # using pool and an extra processor to make sure we get rid of chunk data, which causes problems
        dataset.to_json(hf_file, orient='records', lines=True, force_ascii=False)
        if not self.debug:  # don't compress if debug to save time
            self.logger.info(f"Compressing {split} dataset at {hf_file}")
            os.system(f'xz -zkf -T0 {hf_file}')  # -TO to use multithreading

    def get_df(self, engine, data_to_load: dict, court_string="CH_BGer", use_cache=True, overwrite_cache=False):
        """
        get dataframe of all cases and add additional information such as judgments, sections, file_number, citations
        :param engine:          engine used for db connection
        :param data_to_load:    a dict of booleans specifying which data to load
        :param court_string:    defines which court to load data from
        :param overwrite_cache: whether to load the data from the cache if it exists or whether to load it anew from the db
        :return:                dataframe with all data
        """
        if use_cache:
            # The chunksize is part of the path to distinguish between debug and full datasets
            cache_file = self.data_dir / '.cache' / self.dataset_name / f'{court_string}_{self.get_chunksize()}.parquet.gzip'
            # if cached just load it from there
            if not overwrite_cache:
                df = retrieve_from_cache_if_exists(cache_file)
                if not df.empty:
                    return df

        # otherwise query it from the database
        self.logger.info(f"Retrieving the data from the database for court {court_string}")

        df = self.load_decision(court_string, engine)
        if df.empty:
            self.logger.info(f"Did not find any decisions. Skipping court {court_string}")
            return df  # return right away so we don't run into errors

        df.rename(columns={'lang': 'language'}, inplace=True)
        decision_ids = ["'" + str(x) + "'" for x in df['decision_id'].tolist()]

        if data_to_load['section']:
            df = self.load_section(df, engine)
        if data_to_load['file']:
            df = self.load_file(df, engine)
        if data_to_load['file_number']:
            df = self.load_file_number(df, engine)
        if data_to_load['judgment']:
            df = self.load_judgment(df, engine)
        if data_to_load['citation']:
            df = self.load_citation(decision_ids, df, engine)
        if data_to_load['lower_court']:
            df = self.load_lower_court(decision_ids, df, engine)
        if data_to_load['law_area']:
            df = self.add_law_area(df)
        if data_to_load['law_sub_area']:
            df = self.add_law_sub_area(df)

        df.drop_duplicates(subset=self.get_feature_col_names(), inplace=True)
        self.logger.info("Finished loading the data from the database")
        if use_cache:
            save_df_to_cache(df, cache_file)

        # log col names
        self.logger.info(f"Columns: {df.columns.tolist()}")
        return df

    def load_decision(self, court_string, engine):
        self.logger.info("Loading Decision")
        table = 'decision d LEFT JOIN language ON language.language_id = d.language_id'
        columns = 'd.*, extract(year from d.date) as year, language.iso_code as lang'
        where = f"d.decision_id IN {where_string_court('decision_id', court_string)}"
        return next(self.select(engine, table, columns, where, chunksize=self.get_chunksize()), pd.DataFrame())

    def load_file_number(self, df, engine):
        decision_ids = ["'" + str(x) + "'" for x in df['decision_id'].tolist()]

        self.logger.info('Loading File Number')
        table = f"{join_tables_on_decision(['file_number'])}"
        where = f"file_number.decision_id IN ({','.join(decision_ids)})"
        columns = 'file_number.decision_id, file_number'
        file_number_df = next(self.select(engine, table, columns, where, None, self.get_chunksize()),
                              pd.DataFrame())
        if file_number_df.empty:
            return df

        def extract_file_number(x):
            """Extract the file number from the weird file number string"""
            # Split the string on ','
            parts = x.split(',')
            # Get the second part (the file number) and remove the quotes
            file_number = parts[1].replace('"', '')
            # replace {}() with ''
            file_number = file_number.replace("{", "").replace("}", "").replace("(", "").replace(")", "")
            # Replace spaces with underscores
            file_number = file_number.replace(' ', '_')
            return file_number

        file_number_df['file_number'] = file_number_df['file_number'].map(extract_file_number)
        # cast every element in file_number_df['decision_id'] to string
        file_number_df['decision_id'] = file_number_df['decision_id'].astype(str)
        df = pd.merge(df, file_number_df, left_on='decision_id', right_on='decision_id', how='left')
        return df

    def load_section(self, df, engine, decision_id_col_name='decision_id', origin=False):
        # TODO this could probably be sped up if we just load the sections we need
        self.logger.info('Loading Section')

        # add column 'sections' to df
        df['sections'] = None

        decision_ids = ["'" + str(x) + "'" for x in df[decision_id_col_name].tolist() if x is not None and str(x) != 'nan']
        table = f"{join_tables_on_decision(['num_tokens'])}"
        where = f"section.decision_id IN ({','.join(decision_ids)})"
        section_df = next(self.select(engine, table, "sections, section.decision_id", where, None, self.get_chunksize()), pd.DataFrame())

        for index, row in tqdm(section_df.iterrows()):
            decision_id = str(row['decision_id'])
            section = row['sections']
            # Find the index of the row in df with the matching decision_id
            df[decision_id_col_name] = df[decision_id_col_name].astype(str)
            i = df.index[df[decision_id_col_name] == decision_id].tolist()[0]
            # Set the value of the 'sections' field at the index to the judgments
            df.at[i, 'sections'] = section

        for feature_col in self.get_feature_col_names():
            df = self.expand_df(df, feature_col, origin)
        df.drop(columns=['sections'], inplace=True)

        if not origin:
            df['chamber'] = df.chamber_id.apply(self.get_string_value, args=[self.chamber_dict])  # chamber
            df['court'] = df.chamber.apply(get_court_from_chamber)  # court: first two parts of chamber_string
            df['canton'] = df.chamber.apply(get_canton_from_chamber)  # canton: first part of chamber_string
            df['region'] = df.canton.apply(get_region)

            # drop rows where all the feature cols are nan
            df.dropna(subset=self.get_feature_col_names(), how='all', inplace=True)
        return df

    def get_feature_col_names(self):
        return [feature_col.name.lower() for feature_col in self.feature_cols]

    def get_filter_col_names(self):
        return [filter_col.name.lower() for filter_col in self.filter_cols]

    def load_citation(self, decision_ids, df, engine):
        self.logger.info('Loading Citation')
        table = f"{join_tables_on_decision(['citation'])}"
        where = f"citation.decision_id IN ({','.join(decision_ids)})"
        citations_df = next(self.select(engine, table, "citation.decision_id as decision_id, citations", where, None, self.get_chunksize()), pd.DataFrame())
        if not citations_df.empty:
            citations_df['decision_id'] = citations_df['decision_id'].astype(str)
            df['decision_id'] = df['decision_id'].astype(str)
            df = pd.merge(df, citations_df, left_on='decision_id', right_on='decision_id', how='left')
        else:
            df['citations'] = ""
        return df

    def load_file(self, df, engine):
        self.logger.info('Loading File')
        table = f"{join_tables_on_decision(['file'])}"
        columns = 'file.file_name, file.html_url, file.pdf_url, file.file_id'
        file_ids = ["'" + str(x) + "'" for x in df['file_id'].tolist()]
        if len(file_ids) > 0:
            where = f"file.file_id IN ({','.join(file_ids)})"
            file_df = next(self.select(engine, table, columns, where, None, self.get_chunksize()))
            assert len(file_df) == len(df)
            # iterate through file_df and set the respective values in df
            for index, row in file_df.iterrows():
                file_id = row['file_id']
                file_name = row['file_name']
                html_url = row['html_url']
                pdf_url = row['pdf_url']
                df.loc[df['file_id'] == file_id, 'file_name'] = file_name
                df.loc[df['file_id'] == file_id, 'html_url'] = html_url
                df.loc[df['file_id'] == file_id, 'pdf_url'] = pdf_url
        else:
            self.logger.info("file_ids empty")
        return df

    def load_judgment(self, df, engine):
        self.logger.info('Loading Judgments')
        decision_ids = ["'" + str(x) + "'" for x in df['decision_id'].tolist()]

        table = f"{join_tables_on_decision(['judgment'])}"
        where = f"judgment_map.decision_id IN ({','.join(decision_ids)})"
        judgments_df = next(self.select(engine, table, "judgments, judgment_map.decision_id", where, None, self.get_chunksize()), pd.DataFrame())

        if not judgments_df.empty:
            # add empty column 'judgments' to df
            df['judgments'] = ""

            # iterate through judgments_df and add judgments to df
            for index, row in judgments_df.iterrows():
                decision_id = str(row['decision_id'])
                judgments = row['judgments']
                # Find the index of the row in df with the matching decision_id
                i = df.index[df['decision_id'] == decision_id].tolist()[0]
                # Set the value of the 'judgments' field at the index to the judgments
                df.at[i, 'judgments'] = judgments
        else:
            df['judgments'] = ""
            self.logger.info("judgments_df is empty")
        return df

    def load_lower_court(self, decision_ids, df, engine):
        self.logger.info('Loading Lower Court')
        table = f"{join_tables_on_decision(['lower_court'])}"
        columns = ("lower_court.decision_id as decision_id,"
                   "lower_court.date as origin_date,"
                   "lower_court.court_id as origin_court, "
                   "lower_court.canton_id as origin_canton, "
                   "lower_court.chamber_id as origin_chamber, "
                   "lower_court.file_number as origin_file_number")
        where = f"lower_court.decision_id IN ({','.join(decision_ids)})"
        lower_court_df = next(self.select(engine, table, columns, where, None, self.get_chunksize()), pd.DataFrame())
        if not lower_court_df.empty:
            lower_court_df['decision_id'] = lower_court_df['decision_id'].astype(str)
            df['decision_id'] = df['decision_id'].astype(str)
            df = pd.merge(df, lower_court_df, left_on='decision_id', right_on='decision_id', how='left')

            # load origin_decision_id, origin_facts and origin_considerations
            df = self.load_lower_court_description(df, engine)
        return df

    def load_lower_court_description(self, df, engine):
        origin_file_numbers = ["'" + str(x) + "'" for x in df['origin_file_number'].tolist() if x is not None and str(x) != 'nan']
        if len(origin_file_numbers) == 0:
            return df

        # loading decision_id's of these origin_file_numbers
        table = f"public.file_number"
        columns = "text, decision_id as origin_decision_id"
        where = f"text IN ({','.join(origin_file_numbers)})"
        origin_decision_ids_df = next(self.select(engine, table, columns, where, None, self.get_chunksize()), pd.DataFrame())

        if origin_decision_ids_df.empty:
            self.logger.info("origin_decision_ids_df is empty")
            return df

        # iterate through origin_decision_ids_df and add origin_decision_id to df
        for index, row in origin_decision_ids_df.iterrows():
            origin_file_number = row['text']
            origin_decision_id = row['origin_decision_id']
            # Find the index of the row in df with the matching origin_file_number
            i = df.index[df['origin_file_number'] == origin_file_number].tolist()[0]
            # Set the value of the 'origin_decision_id' field at the index to the origin_decision_id
            df.at[i, 'origin_decision_id'] = origin_decision_id

        df = self.load_section(df, engine, 'origin_decision_id', origin=True)

        return df
    @staticmethod
    def add_law_area(df):
        """ Make law area label using the chamber name """
        # load law area labels
        with open(os.path.join(ROOT_DIR, 'legal_info/chamber_to_area.json'), 'r') as f:
            chamber_to_area_dict = json.load(f)
        # add law area labels
        df['law_area'] = df['chamber'].map(chamber_to_area_dict)
        return df

    @staticmethod
    def add_law_sub_area(df):
        """ Make law sub area label using the chamber name """
        # load law area labels
        with open(os.path.join(ROOT_DIR, 'legal_info/chamber_to_sub_area.json'), 'r') as f:
            chamber_to_sub_area_dict = json.load(f)

        def chamber_to_sub_area(chamber):
            if chamber in chamber_to_sub_area_dict:
                return chamber_to_sub_area_dict[chamber]
            else:
                return np.nan

        # add law area labels
        df['law_sub_area'] = df['chamber'].map(chamber_to_sub_area)
        # make sure that the law_sub_area is a string
        df['law_sub_area'] = df['law_sub_area'].astype(str)
        return df

    @staticmethod
    def get_string_value(x, info_dict):
        if not math.isnan(float(x)):
            return info_dict[int(x)]
        else:
            return np.nan

    def expand_df(self, df, feature_col, origin):
        """
        remove not usable values from dataframe, add num_tokens for each feature_col
        :param df:      dataframe containing all the data
        :param feature_col:  specifying column (=feature_col) which is cleaned
        :param origin:  specifying if the feature_col is from the origin (e.g. origin_facts)
        :return:        dataframe
        """

        if origin:
            feature_col_name = 'origin_' + feature_col
        else:
            feature_col_name = feature_col

        # replace empty and whitespace strings with nan so that they can be removed
        def filter_column(row, section_attr):
            if not isinstance(row, str) and not isinstance(row, list):
                return np.nan
            if isinstance(row, str):
                row = ast.literal_eval(row)  # convert string to list of dicts
            for section in row:
                if section['name'] == feature_col:
                    return section[section_attr]

        df[feature_col_name] = df['sections'].apply(filter_column, section_attr='section_text')

        df[f"{feature_col_name}_num_tokens_bert"] = df['sections'].apply(filter_column, section_attr='num_tokens_bert')
        df[f"{feature_col_name}_num_tokens_bert"] = df[f"{feature_col_name}_num_tokens_bert"].fillna(value=0).astype(int)
        df[f"{feature_col_name}_num_tokens_spacy"] = df['sections'].apply(filter_column, section_attr='num_tokens_spacy')
        df[f"{feature_col_name}_num_tokens_spacy"] = df[f"{feature_col_name}_num_tokens_spacy"].fillna(value=0).astype(int)

        if self.split_type == "date-stratified":
            df = df.dropna(subset=['year'])  # make sure that each entry has an associated year
            df.year = df.year.astype(int)  # convert from float to nicer int

        return df

    def save_dataset(self, dataset: datasets.Dataset, labels: list, folder: Path,
                     split_type="date-stratified", sub_datasets=False, kaggle=False, save_reports=False):
        """
        creates all the files necessary for a kaggle dataset from a given df
        :param dataset:     the huggingface dataset to save
        :param labels:      list of all the labels
        :param folder:      where to save the files
        :param split_type:  "date-stratified", "random", or "all_train"
        :param sub_datasets:whether or not to create the special sub dataset for testing of biases
        :param kaggle:      whether or not to create the special kaggle dataset
        :param save_reports:whether or not to compute and save reports
        :return:
        """
        # clean df before saving it
        dataset = self.clean_dataset(dataset)
        self.logger.info("start creating splits")
        splits = self.create_splits(dataset, split_type, include_all=save_reports)
        self.save_huggingface_dataset(splits, folder)
        save_csv = self.dataset_name != "criticality_prediction" and self.dataset_name != "doc2doc_ir"
        self.save_splits(splits, labels, folder, save_reports, False)

        if sub_datasets:
            sub_datasets_dict = self.create_sub_datasets(splits, split_type)
            sub_datasets_dir = self.create_dir(folder, 'sub_datasets')
            for category, sub_dataset_category in sub_datasets_dict.items():
                self.logger.info(f"Processing sub dataset category {category}")
                category_dir = self.create_dir(sub_datasets_dir, category)
                for sub_dataset, sub_dataset_splits in sub_dataset_category.items():
                    sub_dataset_dir = self.create_dir(category_dir, sub_dataset)
                    self.save_splits(sub_dataset_splits, labels, sub_dataset_dir, save_csvs=[Split.TEST.value])

        if kaggle:
            # save special kaggle files
            kaggle_splits = self.prepare_kaggle_splits(splits)
            kaggle_dir = self.create_dir(folder, 'kaggle')
            self.save_splits(kaggle_splits, labels, kaggle_dir, save_reports=save_reports)

        self.logger.info(f"Saved dataset files to {folder}")
        return splits

    def clean_dataset(self, dataset):
        # replace empty strings with nan so that they can be removed
        self.logger.info(f"start cleaning")
        dataset = self.filter_by_num_tokens(dataset, col_names=self.get_filter_col_names(), conjuctive=self.delete_row_only_if_all_feature_cols_below_cutoff)

        dataset = dataset.remove_columns(['language_id', 'chamber_id', 'file_id', 'topic'])

        return dataset

    def prepare_kaggle_splits(self, splits):
        self.logger.info("Saving the data in kaggle format")
        # deepcopy splits, so we don't mess with the original dict
        kaggle_splits = copy.deepcopy(splits)
        # create solution file
        kaggle_splits['solution'] = kaggle_splits[Split.TEST.value].drop('text', axis='columns')  # drop text
        # rename according to kaggle conventions
        kaggle_splits['solution'] = kaggle_splits['solution'].rename(columns={"label": "Expected"})
        # create test file
        kaggle_splits[Split.TEST.value] = kaggle_splits[Split.TEST.value].drop('label', axis='columns')  # drop label
        # create sampleSubmission file
        # rename according to kaggle conventions
        sample_submission = kaggle_splits['solution'].rename(columns={"Expected": "Predicted"})
        # set to random value
        sample_submission['Predicted'] = np.random.choice(kaggle_splits['solution']['Expected'],
                                                          size=len(kaggle_splits['solution']))
        kaggle_splits['sample_submission'] = sample_submission
        return kaggle_splits

    def save_splits(self, splits: dict, labels: list, folder: Path,
                    save_reports=True, save_csvs: Union[list, bool] = True):
        """
        Saves the splits to the filesystem and generates reports
        :param splits:          the splits dictionary to be saved
        :param labels:          list of labels to be saved
        :param folder:          where to save the splits
        :param save_reports:    whether to save reports
        :param save_csvs:       whether to save csv files
        :return:
        """
        self.save_labels(labels, folder)
        for split, dataset in splits.items():
            if len(dataset) < 2:
                self.logger.info(f"Skipping split {split} because "
                                 f"{len(dataset)} entries are not enough to create reports.")
                continue
            self.logger.info(f"Processing split {split}")

            if save_csvs:
                # without the feature_cols, the dataset should fit into RAM
                # Additionally, we don't want to save the long text columns to the csv files because it becomes unreadable
                # TODO this line causes out of memory errors for large datasets like criticality or doc2doc
                #  - set save_reports False for those datasets
                df = dataset.remove_columns(self.get_feature_col_names()).to_pandas()
                if isinstance(save_csvs, list):
                    if split not in save_csvs:
                        continue  # Only save if the split is in the list
                self.logger.info("Saving csv file")
                df.to_csv(folder / f"{split}.csv", index_label='id', index=False)
            if save_reports:
                self.logger.info(f"Computing metadata reports")
                self.save_report(folder, split, dataset)

    def create_splits(self, dataset, split_type, include_all=False):
        # TODO is the .value of the Split enum really necessary? It probably also works without
        self.logger.info(f"Dividing data into splits based on split_type: {split_type}")
        if split_type == "random":
            train, val, test = self.split_random(dataset)
            splits = {Split.TRAIN.value: train, Split.VALIDATION.value: val, Split.TEST.value: test}
        elif split_type == "date-stratified":
            train, val, test = self.split_date_stratified(dataset, self.start_years)
            splits = {Split.TRAIN.value: train, Split.VALIDATION.value: val, Split.TEST.value: test}
        elif split_type == "all_train":
            splits = {Split.TRAIN.value: dataset}  # no split at all
        else:
            raise ValueError("Please supply a valid split_type")
        if include_all:
            # we need to update it since some entries have been removed
            splits[Split.ALL.value] = concatenate_datasets(list(splits.values()))

        return splits

    def create_sub_datasets(self, splits, split_type):
        """
        Creates sub datasets for applications extending beyond the normal splits
        :param split_type:  the type of splitting the data (date-stratified or random)
        :param splits:      the dictionary containing the split dataframes
        :return:
        """
        self.logger.info("Creating sub datasets")
        # TODO debug this

        # set up data structure
        sub_datasets_dict = {metadata: dict() for metadata in self.metadata}
        sub_datasets_dict['input_length'] = dict()

        self.logger.info(f"Processing sub dataset input_length")
        boundaries = [0, 512, 1024, 2048, 4096, 8192]
        for i in range(len(boundaries) - 1):
            lower, higher = boundaries[i] + 1, boundaries[i + 1]
            sub_dataset = sub_datasets_dict['input_length'][f'between({lower:04d},{higher:04d})'] = dict()
            for split_name, split_df in splits.items():
                sub_dataset[split_name] = split_df[split_df.num_tokens_bert.between(lower, higher)]

        self.logger.info(f"Processing sub dataset year")
        if split_type == "date-stratified":
            for year in range(self.start_years[Split.TEST.value], self.current_year):
                sub_dataset = sub_datasets_dict['year'][str(year)] = dict()
                for split_name, split_df in splits.items():
                    sub_dataset[split_name] = split_df[split_df.year == year]

        self.logger.info(f"Processing sub dataset legal_area")
        for legal_area in legal_areas.keys():
            sub_dataset = sub_datasets_dict['legal_area'][legal_area] = dict()
            for split_name, split_df in splits.items():
                sub_dataset[split_name] = split_df[split_df.legal_area.astype('str').str.contains(legal_area)]

        self.logger.info(f"Processing sub dataset origin_region")
        for region in splits[Split.ALL.value].origin_region.dropna().unique().tolist():
            sub_dataset = sub_datasets_dict['origin_region'][region] = dict()
            for split_name, split_df in splits.items():
                region_df = split_df.dropna(subset=['origin_region'])
                sub_dataset[split_name] = region_df[region_df.origin_region.astype('str').str.contains(region)]

        self.logger.info(f"Processing sub dataset origin_canton")
        for canton in splits[Split.ALL.value].origin_canton.dropna().unique().tolist():
            sub_dataset = sub_datasets_dict['origin_canton'][canton] = dict()
            for split_name, split_df in splits.items():
                canton_df = split_df.dropna(subset=['origin_canton'])
                sub_dataset[split_name] = canton_df[canton_df.origin_canton.astype('str').str.contains(canton)]

        self.logger.info(f"Processing sub dataset origin_court")
        for court in splits[Split.ALL.value].origin_court.dropna().unique().tolist():
            sub_dataset = sub_datasets_dict['origin_court'][court] = dict()
            for split_name, split_df in splits.items():
                court_df = split_df.dropna(subset=['origin_court'])
                sub_dataset[split_name] = court_df[court_df.origin_court.astype('str').str.contains(court)]

        self.logger.info(f"Processing sub dataset origin_chamber")
        for chamber in splits[Split.ALL.value].origin_chamber.dropna().unique().tolist():
            sub_dataset = sub_datasets_dict['origin_chamber'][chamber] = dict()
            for split_name, split_df in splits.items():
                chamber_df = split_df.dropna(subset=['origin_chamber'])
                sub_dataset[split_name] = chamber_df[chamber_df.origin_chamber.astype('str').str.contains(chamber)]

        return sub_datasets_dict

    def save_report(self, folder, split, dataset):
        """
        Saves statistics about the dataset in the form of csv tables and png graphs.
        :param folder:  the base folder to save the report to
        :param split:   the name of the split
        :param df:      the df containing the dataset
        :return:
        """
        self.logger.info(f"Saving report for split {split}")
        split_folder = self.create_dir(self.reports_folder, f'{split}')
        disable_pandas = (self.dataset_name == "doc2doc_ir") or (self.dataset_name == "criticality_prediction") or (self.dataset_name == "citation_extraction")
        report_creator = ReportCreator(split_folder, self.debug, disable_pandas)
        report_creator.report_general(self.metadata, self.get_feature_col_names(), self.labels, dataset)
        if not disable_pandas:
            # self.plot_custom(report_creator, dataset, split_folder)
            pass

    @abc.abstractmethod
    def plot_custom(self, report_creator, dataset, folder):
        # TODO change df to dataset for all plot_custom
        """
        Implement custom plots for each dataset_creator in this method
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def save_labels(self, labels, folder):
        """
        Saves the labels and the corresponding ids as a json file
        :param labels:      list of labels dict
        :param folder:      where to save the labels
        :return:
        """
        if labels:  # labels can also be None (for PretrainingDatasetCreator), in which case we do nothing
            assert len(labels) <= 2
            i = 1
            for entry in labels:
                entry = list(entry)
                labels_dict = dict(enumerate(entry))
                json_labels = {"id2label": labels_dict, "label2id": {y: x for x, y in labels_dict.items()}}
                if len(labels) != 1:
                    file_name = folder / f"labels_{i}.json"
                    i = i + 1
                else:
                    file_name = folder / "labels.json"
                if not os.path.isdir(folder):
                    os.mkdir(folder)
                with open(f"{file_name}", 'w', encoding='utf-8') as f:
                    json.dump(json_labels, f, ensure_ascii=False, indent=4)
        else:
            self.logger.info("No labels given.")

    def split_date_stratified(self, dataset, start_years: dict):
        """
        Splits the dataset into train, val and test based on the date
        :param dataset:            the dataset to be split
        :param start_years:   the years when to start each split
        :return:
        """
        train = dataset.filter(
            lambda x: x["year"] in range(start_years[Split.TRAIN.value], start_years[Split.VALIDATION.value]))
        val = dataset.filter(
            lambda x: x["year"] in range(start_years[Split.VALIDATION.value], start_years[Split.TEST.value]))
        test = dataset.filter(
            lambda x: x["year"] in range(start_years[Split.TEST.value], self.current_year+2))
        return train, val, test

    def split_random(self, dataset):
        """
        Splits the dataset randomly into train, val and test
        :param dataset:      the dataset to be split
        :return:
        """
        # 80% train, 20% test + validation
        train_testvalid = dataset.train_test_split(test=0.2)
        # Split the 20% test + valid in half test, half valid
        test_valid = train_testvalid[Split.TEST.value].train_test_split(test=0.5)
        # gather everything into a single DatasetDict
        return train_testvalid[Split.TRAIN.value], test_valid[Split.TRAIN.value], test_valid[Split.TEST.value]

    def create_overview(self, path=None, export_path=None, export_name="overview", include_all=False, parallel=True):
        """
        :function:              creates an overview of the dataset
        :param path:            path to the court dataset folders
        :param export_name:     name of the exported file without extension
        :param export_path:     path to the folder where the file should be exported
        :param include_all:     if True, all courts are included in the overview otherwise only the created courts
        :param parallel:        if True, the overview is created in parallel processing
        """
        if path is None:
            path = self.get_dataset_folder()
        if export_path is None:
            export_path = self.get_dataset_folder()

        self.logger.info("Creating overview of the datasets")
        courts_av_tmp = os.listdir(path)
        courts_av = []
        # filtering to only have dir's
        for s in courts_av_tmp:
            if not os.path.isfile(f"{path}/{s}"):
                courts_av.append(s)

        #   stores the overview in a list of dicts
        courts_data = []

        if parallel:
            self.logger.info("Creating overview in parallel")
            with Pool(processes=self.num_cores) as pool:
                results = [pool.apply_async(get_court_len, (court, path)) for court in tqdm(courts_av)]
                for result in results:
                    courts_data.append(result.get())
            pool.close()
        else:
            self.logger.info("Creating overview sequentially")
            for court in tqdm(courts_av):
                court_data = get_court_len(court, path)
                courts_data.append(court_data)

        # add courts that are not in the folder
        if include_all:
            for court in self.get_all_courts():
                if court not in courts_av:
                    court_data = {"name": court, 'created': False}
                    courts_data.append(court_data)

        # check if export file already exists and increment the name index if it does
        version = 1
        while os.path.exists(f"{export_path}/{export_name}_v{version}.csv"):
            version += 1
        export_name = f"{export_name}_v{version}.csv"

        # export to csv
        with open(os.path.join(export_path, export_name), "w") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "all", Split.VALIDATION.value, Split.TEST.value,
                                                   Split.TRAIN.value, Split.SECRET_TEST.value, "created"])
            writer.writeheader()
            writer.writerows(courts_data)
        self.logger.info(f"Overview created and exported to: {os.path.join(export_path, export_name)}")


    # function which takes col name as input and filters df/dataset by number of token
    def filter_by_num_tokens(self, data_structure, col_names, court=None, conjuctive=False):
        """
        :function:              filters df / dataset by the number of tokens in the column 'col_name'
        :param data_structure:  pandas dataframe or huggingface dataset
        :param col_name:        name of the column to filter by (e.g. "origin_facts")
        :param conjuctive:      if True, deletes only rows where all columns have less than the cutoff
        :return:                filtered df / dataset
        """
        if len(col_names) > 1:
            if conjuctive:
                self.logger.info("Deleting only rows where all columns have less than the cutoff")
            else:
                self.logger.info("Deleting rows where any column has less than the cutoff")
        # TODO: save the cutoffs in a json file and load them from there
        facts_cutoff = {
            "VD_TC": 310,
            "CH_BGer": 250,
            "SO_VSG": 200,
            "ZH_OG": 300,
            "TI_TRAC": 120,
            "TI_PP": 140,
            "GE_CJ": 170,
            "SO_OG": 300,
            "SO_VG": 150,
            "ZH_KSG": 150,
            "BS_APG": 170,
            "CH_BVGE": 140,
            "GL_VG": 150,
            "BL_KG": 500
        }
        considerations_cutoff = {
            "VD_TC": 50,
            "TI_TRAC": 200,
            "CH_BGer": 150,
            "CH_BVGE": 200,
            "GE_CJ": 50,
            "GR_KG": 150,
            "TI_TCAS": 150,
            "TI_TE": 150,
            "TI_TRAP": 150,
            "FR_TC": 100,
            "TI_TCA": 150,
            "TI_PP": 150,
            "JU_TC": 150
        }

        def get_cutoff(col_name, court):
            if col_name == 'facts' or col_name == 'origin_facts':
                return facts_cutoff.get(court, 200) # default cutoff is 200
            elif col_name == 'considerations' or col_name == 'origin_considerations':
                return considerations_cutoff.get(court, 500) # default cutoff is 500
            elif col_name == 'rulings' or col_name == 'origin_rulings':
                return 100  # default cutoff is 100
            else:
                raise ValueError(f"{col_name} not implmemented: col_name must be 'rulings', 'origin_rulings', 'facts', 'origin_facts', "
                                 f"'considerations' or 'origin_considerations'")

        def is_above_cutoff(x):
            if conjuctive:  # returns True if any column is above the cutoff
                for col_name in col_names:
                    if x[col_name + '_num_tokens_bert'] >= get_cutoff(col_name, x['court']):
                        return True
                return False
            else: # returns False if any column is below the cutoff
                for col_name in col_names:
                    if x[col_name + '_num_tokens_bert'] < get_cutoff(col_name, x['court']):
                        return False
                return True

        # check if data_structure is a dataframe or a dataset
        if isinstance(data_structure, pd.DataFrame):
            if len(col_names) > 1:
                raise NotImplementedError("If data_structure is a dataframe, only one column can be filtered")
            col_name = col_names[0]
            self.logger.info(f"filtering {col_name} by num_tokens")
            if court is None:
                raise ValueError("If data_structure is a dataframe, court must be given")
            data_structure = data_structure.loc[data_structure[col_name + '_num_tokens_bert'] >= get_cutoff(col_name, court)]
        elif isinstance(data_structure, datasets.Dataset):
            self.logger.info(f"filtering {','.join(col_names)} by num_tokens")
            data_structure = data_structure.filter(is_above_cutoff)
        else:
            raise ValueError(f"data_structure must be a dataframe or a dataset. Got {type(data_structure)}")

        return data_structure

    def parallelize_dataframe(self, df, func, n_cores=10):
        df_split = np.array_split(df, n_cores)
        pool = Pool(n_cores)
        tmp = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return tmp


def get_court_len(court, path):
    """
    :function:   part of the create_overview function: Gets the number of samples for each split of a court
    :param court:   name of the court
    :param path:    path to the dataset folder
    :return:        dict with the number of samples for each split of a court
    """

    court_data = {"name": court}
    for key in [split.value for split in Split]:  # ["all", "val", "test", "train", "secret_test"]
        try:
            with open(os.path.join(path, court, f"{key}.csv"), "r") as f:
                reader = csv.reader(f)
                court_data[key] = len(list(reader)) - 1  # -1 because of header
        except FileNotFoundError:
            court_data[key] = -2
    court_data['created'] = True
    return court_data
