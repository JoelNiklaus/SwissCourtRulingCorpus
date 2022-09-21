import abc
import copy
import math
from collections import Counter
from pathlib import Path
from typing import Union
import ast
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

import dask.dataframe as dd
import numpy as np
import pandas as pd
from scrc.enums.section import Section

from scrc.preprocessors.abstract_preprocessor import AbstractPreprocessor
from scrc.utils.log_utils import get_logger
import json
from scrc.utils.main_utils import retrieve_from_cache_if_exists, save_df_to_cache

from scrc.utils.sql_select_utils import get_legal_area, join_tables_on_decision, legal_areas, get_region, \
    select_sections_with_decision_and_meta_data, where_string_spider

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

    def __init__(self, config: dict):
        __metaclass__ = abc.ABCMeta
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.seed = 42
        self.minFeatureColLength = 100  # characters
        self.debug_chunksize = 100
        self.real_chunksize = 1_000_000

        self.split_type = None  # to be overridden
        self.dataset_name = None  # to be overridden
        self.feature_cols = ["text"]  # to be overridden

    @abc.abstractmethod
    def get_dataset(self, feature_col, save_reports):
        pass

    def get_chunksize(self):
        if self.debug:
            return int(self.debug_chunksize)  # run on smaller dataset for testing
        else:
            return int(self.real_chunksize)

    def create_dataset(self, sub_datasets=False, kaggle=False, huggingface=False, save_reports=False):
        """
        Retrieves the respective function named by the dataset and executes it to get the df for that dataset.
        :return:
        """
        self.logger.info(f"Creating {self.dataset_name} dataset")

        dataset_folder = self.create_dir(self.datasets_subdir, self.dataset_name)

        processed_file_path = self.progress_dir / f"dataset_{self.dataset_name}_created.txt"
        datasets, message = self.compute_remaining_parts(processed_file_path, ["-".join(self.feature_cols)])
        self.logger.info(message)

        # Check these todos on the judgment_dataset_creator
        # TODO not one dataset per feature col, but put all feature cols always as lists of paragraphs (facts, considerations, etc.) into the same dataset
        # TODO put all languages into the same dataset
        # TODO check if all columns are fetched correctly (num_tokens_bert, etc.)

        if datasets:
            feature_cols = datasets
            feature_col_folder = self.create_dir(dataset_folder, "-".join(feature_cols))

            df, labels = self.get_dataset(feature_cols, save_reports)
            df = df.sample(frac=1).reset_index(drop=True)  # shuffle dataset to make sampling easier
            splits = self.save_dataset(df, labels, feature_col_folder, self.split_type,
                                       sub_datasets=sub_datasets, kaggle=kaggle, save_reports=save_reports)

            if huggingface:
                self.logger.info("Generating huggingface dataset")
                self.save_huggingface_dataset(splits, feature_col_folder)

            self.mark_as_processed(processed_file_path, feature_cols)
        else:
            self.logger.info("All parts have been computed already.")

    def save_huggingface_dataset(self, splits, feature_col_folder):
        """
        save data as huggingface dataset with columns: 'id', 'year': year, 'language',
        region', 'canton', 'legal area', 'bge_label', 'considerations' and 'facts'
        ATTENTION: only works for feature_cols = [considerations, facts]
        :param splits:                  specifying splits of dataset
        :param feature_col_folder:      name of folder
        """
        huggingface_dir = self.create_dir(feature_col_folder, 'huggingface')
        for split in ['train', 'val', 'test', 'secret_test']:
            records = []
            df = splits[split]
            i = 0
            # ATTENTION this works only for 1 or 2 entries in feature_col
            if len(self.feature_cols) > 1:
                i = 1
            tuple_iterator = zip(df.index, df['year'], df['legal_area'], df['origin_region'],
                                 df['origin_canton'], df['label'], df['lang'], df[self.feature_cols[0]],
                                 df[self.feature_cols[i]])

            for case_id, year, legal_area, region, canton, label, lang, a, b in tuple_iterator:
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
                    'label': label,
                }
                record[self.feature_cols[0]] = a
                if i == 1:
                    record[self.feature_cols[1]] = b

                records.append(record)
            with open(f'{huggingface_dir}/{split}.jsonl', 'w') as out_file:
                for record in records:
                    out_file.write(json.dumps(record) + '\n')

    def get_df(self, engine, feature_col):

        cache_dir = self.data_dir / '.cache' / f'{self.dataset_name}_{self.get_chunksize()}.csv'
        df = retrieve_from_cache_if_exists(cache_dir)
        if df.empty:
            self.logger.info("Retrieving the data from the database")

            where_string = f"d.decision_id IN {where_string_spider('decision_id', 'CH_BGer')}"
            table_string = 'decision d LEFT JOIN language ON language.language_id = d.language_id'
            decision_df = next(
                self.select(engine, table_string, 'd.*, extract(year from d.date) as year, language.iso_code as lang',
                            where_string,
                            chunksize=self.get_chunksize()))
            decision_ids = ["'" + str(x) + "'" for x in decision_df['decision_id'].tolist()]

            print('Loading Judgments')
            table = f"{join_tables_on_decision(['judgment'])}"
            where = f"judgment_map.decision_id IN ({','.join(decision_ids)})"
            judgments_df = next(self.select(engine, table, "judgments", where, None, self.get_chunksize()))
            decision_df['judgments'] = judgments_df['judgments']

            print('Loading File')
            table = f"{join_tables_on_decision(['file'])}"
            file_ids = ["'" + str(x) + "'" for x in decision_df['file_id'].tolist()]
            where = f"file.file_id IN ({','.join(file_ids)})"
            file_df = next(self.select(engine, table, 'file.file_name, file.html_url, file.pdf_url', where, None,
                                       self.get_chunksize()))
            decision_df['file_name'] = file_df['file_name']
            decision_df['html_url'] = file_df['html_url']
            decision_df['pdf_url'] = file_df['pdf_url']

            print('Loading Lower Court')
            table = f"{join_tables_on_decision(['lower_court'])}"
            where = f"lower_court.decision_id IN ({','.join(decision_ids)})"
            lower_court_select_fields = ("lower_court.date as origin_date,"
                                         "lower_court.court_id as origin_court, "
                                         "lower_court.canton_id as origin_canton, "
                                         "lower_court.chamber_id as origin_chamber, "
                                         "lower_court.file_number as origin_file_number")
            lower_court_df = next(
                self.select(engine, table, lower_court_select_fields, where, None, self.get_chunksize()))
            decision_df['origin_date'] = lower_court_df['origin_date']
            decision_df['origin_court'] = lower_court_df['origin_court']
            decision_df['origin_canton'] = lower_court_df['origin_canton']
            decision_df['origin_chamber'] = lower_court_df['origin_chamber']
            decision_df['origin_file_number'] = lower_court_df['origin_file_number']

            print('Loading Citation')
            table = f"{join_tables_on_decision(['citation'])}"
            where = f"citation.decision_id IN ({','.join(decision_ids)})"
            citations_df = next(self.select(engine, table, "citations", where, None, self.get_chunksize()))
            decision_df['citations'] = citations_df['citations']

            print('Loading Section')
            table = f"{join_tables_on_decision(['num_tokens'])}"
            where = f"section.decision_id IN ({','.join(decision_ids)})"
            section_df = next(self.select(engine, table, "sections", where, None, self.get_chunksize()))
            decision_df['sections'] = section_df['sections']

            print('Loading File Number')
            table = f"{join_tables_on_decision(['file_number'])}"
            where = f"file_number.decision_id IN ({','.join(decision_ids)})"
            file_number_df = next(self.select(engine, table, "file_numbers", where, None, self.get_chunksize()))
            # we get a list of file_numbers but only want one, all entries are the same but different syntax
            def get_one_file_number(column_data):
                file_number = str(next(iter(column_data or []), None))
                file_number = file_number.replace(" ", "_")
                file_number = file_number.replace(".", "_")
                return file_number
            decision_df['file_number'] = file_number_df['file_numbers'].map(get_one_file_number)

            save_df_to_cache(decision_df, cache_dir)
            df = decision_df
        for feature_col in list(feature_col)[0].split('-'):
            df = self.clean_df(df, feature_col)
        df['legal_area'] = df.chamber_id.apply(get_legal_area)
        df['origin_region'] = df.origin_canton.apply(get_region)

        self.logger.info("Finished loading the data from the database")
        return df

    def clean_df(self, df, column):
        # replace empty and whitespace strings with nan so that they can be removed
        sections = df['sections']

        def filter_column(column_data):
            if not isinstance(column_data, str) and not isinstance(column_data, list): return np.nan
            if isinstance(column_data, str):
                column_data = ast.literal_eval(column_data)
            for section in column_data:
                if section['name'] == column:
                    return section['section_text']

        df[column] = sections.map(filter_column)
        df[column] = df[column].replace(r'^\s+$', np.nan, regex=True)
        df[column] = df[column].replace('', np.nan)

        def filter_column(column_data):
            if not isinstance(column_data, str) and not isinstance(column_data, list): return np.nan
            if isinstance(column_data, str):
                column_data = ast.literal_eval(column_data)
            for section in column_data:
                if section['name'] == column:
                    return section['num_tokens_bert']
        df[f"{column}_num_tokens_bert"] = sections.map(filter_column)

        def filter_column(column_data):
            if not isinstance(column_data, str) and not isinstance(column_data, list): return np.nan
            if isinstance(column_data, str):
                column_data = ast.literal_eval(column_data)
            for section in column_data:
                if section['name'] == column:
                    return section['num_tokens_spacy']
        df[f"{column}_num_tokens_spacy"] = sections.map(filter_column)

        # TODO do this only if that's the case for all feature_cols!!
        if self.split_type == "date-stratified":
            df = df.dropna(subset=['year'])  # make sure that each entry has an associated year
        df.year = df.year.astype(int)  # convert from float to nicer int

        # filter out entries where the feature_col (text/facts/considerations) is less than 100 characters
        # because then it is most likely faulty
        # TODO think about including this line
        # df = df[df[column].str.len() > self.minFeatureColLength]
        return df

    def save_dataset(self, df: pd.DataFrame, labels: list, folder: Path,
                     split_type="date-stratified", split=(0.7, 0.1, 0.2),
                     sub_datasets=False, kaggle=False, save_reports=False):
        """
        creates all the files necessary for a kaggle dataset from a given df
        :param df:          needs to contain the columns text and label
        :param labels:      list of all the labels
        :param folder:      where to save the files
        :param split_type:  "date-stratified" or "random"
        :param split:       how to split the data into train, val and test set: needs to sum up to 1
        :param sub_datasets:whether or not to create the special sub dataset for testing of biases
        :param kaggle:      whether or not to create the special kaggle dataset
        :param save_reports:whether or not to compute and save reports
        :return:
        """
        splits = self.create_splits(df, split, split_type, include_all=save_reports)
        self.save_splits(splits, labels, folder, save_reports=save_reports)

        if sub_datasets:
            sub_datasets_dict = self.create_sub_datasets(splits, split_type)
            sub_datasets_dir = self.create_dir(folder, 'sub_datasets')
            for category, sub_dataset_category in sub_datasets_dict.items():
                self.logger.info(f"Processing sub dataset category {category}")
                category_dir = self.create_dir(sub_datasets_dir, category)
                for sub_dataset, sub_dataset_splits in sub_dataset_category.items():
                    sub_dataset_dir = self.create_dir(category_dir, sub_dataset)
                    self.save_splits(sub_dataset_splits, labels, sub_dataset_dir, save_csvs=['test'])

        if kaggle:
            # save special kaggle files
            kaggle_splits = self.prepare_kaggle_splits(splits)
            kaggle_dir = self.create_dir(folder, 'kaggle')
            self.save_splits(kaggle_splits, labels, kaggle_dir, save_reports=save_reports)

        self.logger.info(f"Saved dataset files to {folder}")
        return splits

    def prepare_kaggle_splits(self, splits):
        self.logger.info("Saving the data in kaggle format")
        # deepcopy splits, so we don't mess with the original dict
        kaggle_splits = copy.deepcopy(splits)
        # create solution file
        kaggle_splits['solution'] = kaggle_splits['test'].drop('text', axis='columns')  # drop text
        # rename according to kaggle conventions
        kaggle_splits['solution'] = kaggle_splits['solution'].rename(columns={"label": "Expected"})
        # create test file
        kaggle_splits['test'] = kaggle_splits['test'].drop('label', axis='columns')  # drop label
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
        :param labels:          the labels to be saved
        :param folder:          where to save the splits
        :param save_reports:    whether to save reports
        :param save_csvs:       whether to save csv files
        :return:
        """
        self.save_labels(labels, folder)
        for split, df in splits.items():
            if len(df.index) < 2:
                self.logger.info(f"Skipping split {split} because "
                                 f"{len(df.index)} entries are not enough to create reports.")
                continue
            self.logger.info(f"Processing split {split}")

            if save_reports:
                self.logger.info(f"Computing metadata reports")
                self.save_report(folder, split, df)

            if save_csvs:
                if isinstance(save_csvs, list):
                    if split not in save_csvs:
                        continue  # Only save if the split is in the list
                self.logger.info("Saving csv file")
                df.to_csv(folder / f'{split}.csv', index_label='id', index=False)

    def create_splits(self, df, split, split_type, include_all=False):
        # TODO add secret test for date_stratified
        self.logger.info("Splitting data into train, val and test set")
        if split_type == "random":
            train, val, test = self.split_random(df, split)
            secret_test = pd.DataFrame()
        elif split_type == "date-stratified":
            train, val, test, secret_test = self.split_date_stratified(df, split)
        else:
            raise ValueError("Please supply a valid split_type")
        splits = {'train': train, 'val': val, 'test': test, 'secret_test': secret_test}
        if include_all:
            splits['all'] = pd.concat([train, val, test, secret_test])  # we need to update it since some entries have been removed

        return splits

    def create_sub_datasets(self, splits, split_type):
        """
        Creates sub datasets for applications extending beyond the normal splits
        :param split_type:  the type of splitting the data (date-stratified or random)
        :param splits:      the dictionary containing the split dataframes
        :return:
        """
        self.logger.info("Creating sub datasets")

        # set up data structure
        sub_datasets_dict = {
            'input_length': dict(), 'year': dict(), 'legal_area': dict(),
            'origin_region': dict(), 'origin_canton': dict(), 'origin_court': dict(), 'origin_chamber': dict(),
        }

        self.logger.info(f"Processing sub dataset input_length")
        boundaries = [0, 512, 1024, 2048, 4096, 8192]
        for i in range(len(boundaries) - 1):
            lower, higher = boundaries[i] + 1, boundaries[i + 1]
            sub_dataset = sub_datasets_dict['input_length'][f'between({lower:04d},{higher:04d})'] = dict()
            for split_name, split_df in splits.items():
                sub_dataset[split_name] = split_df[split_df.num_tokens_bert.between(lower, higher)]

        self.logger.info(f"Processing sub dataset year")
        if split_type == "date-stratified":
            for year in range(2017, 2020 + 1):
                sub_dataset = sub_datasets_dict['year'][str(year)] = dict()
                for split_name, split_df in splits.items():
                    sub_dataset[split_name] = split_df[split_df.year == year]

        self.logger.info(f"Processing sub dataset legal_area")
        for legal_area in legal_areas.keys():
            sub_dataset = sub_datasets_dict['legal_area'][legal_area] = dict()
            for split_name, split_df in splits.items():
                sub_dataset[split_name] = split_df[split_df.legal_area.str.contains(legal_area)]

        self.logger.info(f"Processing sub dataset origin_region")
        for region in splits['all'].origin_region.dropna().unique().tolist():
            sub_dataset = sub_datasets_dict['origin_region'][region] = dict()
            for split_name, split_df in splits.items():
                region_df = split_df.dropna(subset=['origin_region'])
                sub_dataset[split_name] = region_df[region_df.origin_region.str.contains(region)]

        self.logger.info(f"Processing sub dataset origin_canton")
        for canton in splits['all'].origin_canton.dropna().unique().tolist():
            sub_dataset = sub_datasets_dict['origin_canton'][canton] = dict()
            for split_name, split_df in splits.items():
                canton_df = split_df.dropna(subset=['origin_canton'])
                sub_dataset[split_name] = canton_df[canton_df.origin_canton.str.contains(canton)]

        self.logger.info(f"Processing sub dataset origin_court")
        for court in splits['all'].origin_court.dropna().unique().tolist():
            sub_dataset = sub_datasets_dict['origin_court'][court] = dict()
            for split_name, split_df in splits.items():
                court_df = split_df.dropna(subset=['origin_court'])
                sub_dataset[split_name] = court_df[court_df.origin_court.str.contains(court)]

        self.logger.info(f"Processing sub dataset origin_chamber")
        for chamber in splits['all'].origin_chamber.dropna().unique().tolist():
            sub_dataset = sub_datasets_dict['origin_chamber'][chamber] = dict()
            for split_name, split_df in splits.items():
                chamber_df = split_df.dropna(subset=['origin_chamber'])
                sub_dataset[split_name] = chamber_df[chamber_df.origin_chamber.str.contains(chamber)]

        return sub_datasets_dict

    def save_report(self, folder, split, df):
        """
        Saves statistics about the dataset in the form of csv tables and png graphs.
        :param folder:  the base folder to save the report to
        :param split:   the name of the split
        :param df:      the df containing the dataset
        :return:
        """
        split_folder = self.create_dir(folder, f'reports/{split}')

        barplot_attributes = ['legal_area', 'origin_region', 'origin_canton', 'origin_court', 'origin_chamber']
        for attribute in barplot_attributes:
            self.plot_barplot_attribute(df, split_folder, attribute)

        for feature_col in self.feature_cols:
            dict = {f'{feature_col}_num_tokens_bert': 'num_tokens_bert',
                    f'{feature_col}_num_tokens_spacy': 'num_tokens_spacy'}
            self.plot_input_length(df.rename(columns=dict), split_folder, feature_col=feature_col)

        self.plot_custom(df, split_folder, folder)


    @staticmethod
    def plot_barplot_attribute(df, split_folder, attribute, label=""):
        """
        Plots the distribution of the attribute of the decisions in the given dataframe
        :param df:              the dataframe containing the legal areas
        :param split_folder:    where to save the plots and csv files
        :param attribute:       the attribute to barplot
        :param label:           defines if only critical data of a label is considered
        :return:
        """
        attribute_df = df[attribute].value_counts().to_frame()
        total = len(df.index)
        # we deleted the ones where we did not find any attribute: also mention them in this table
        uncategorized = total - attribute_df[attribute].sum()
        attribute_df.at['uncategorized', attribute] = uncategorized
        attribute_df.at['all', attribute] = total
        attribute_df = attribute_df.reset_index(level=0)
        attribute_df = attribute_df.rename(columns={'index': attribute, attribute: 'number of decisions'})
        attribute_df['number of decisions'] = attribute_df['number of decisions'].astype(int)
        attribute_df['percent'] = round(attribute_df['number of decisions'] / total, 4)

        temp_df = attribute_df.iloc[-2:]
        attribute_df = attribute_df.iloc[:-2]
        if attribute == 'counter':
            attribute_df[attribute] = attribute_df[attribute].astype(float)
        attribute_df.sort_values(by=[attribute], inplace=True)
        attribute_df = pd.concat([attribute_df, temp_df], axis=0)

        attribute_df.to_csv(split_folder / f'{attribute}_{label}_distribution.csv')
        # need to make sure to use right type
        attribute_df[attribute] = attribute_df[attribute].astype(str)
        attribute_df = attribute_df[~attribute_df[attribute].str.contains('all')]
        fig = px.bar(attribute_df, x=attribute, y="number of decisions", title=f'{attribute}_{label}_distribution-histogram')
        fig.write_image(split_folder / f'{attribute}_{label}_distribution-histogram.png')
        plt.clf()

    @staticmethod
    def plot_labels(df, split_folder, label_name='label'):
        """
        Plots the label distribution of the decisions in the given dataframe
        :param df:              the dataframe containing the labels
        :param split_folder:    where to save the plots and csv files
        :param label_name:      name of the original label
        :return:
        """
        # compute label imbalance
        # ax = df.label.astype(str).hist()
        # ax.tick_params(labelrotation=30)
        # ax.get_figure().savefig(split_folder / 'multi_label_distribution.png', bbox_inches="tight")

        counter_dict = dict(Counter(np.hstack(df.label)))
        counter_dict['all'] = sum(counter_dict.values())
        label_counts = pd.DataFrame.from_dict(counter_dict, orient='index', columns=['num_occurrences'])
        label_counts.loc[:, 'percent'] = round(label_counts['num_occurrences'] / counter_dict['all'], 4)
        label_counts.to_csv(split_folder / f"{label_name}_distribution.csv", index_label='label')

        ax = label_counts[~label_counts.index.str.contains("all")].plot.bar(y='num_occurrences', rot=15)
        ax.get_figure().savefig(split_folder / f"{label_name}_distribution.png", bbox_inches="tight")
        plt.clf()

    @staticmethod
    def plot_input_length(df, split_folder, feature_col='text'):
        """
        Plots the input length of the decisions in the given dataframe
        :param df:              the dataframe containing the decision texts
        :param split_folder:    where to save the plots and csv files
        :param feature_col:     spezifies feature_col
        :return:
        """
        # compute median input length
        input_length_distribution = df[['num_tokens_spacy', 'num_tokens_bert']].describe().round(0).astype(int)
        input_length_distribution.to_csv(split_folder / f'{feature_col}_input_length_distribution.csv', index_label='measure')

        # bin outliers together at the cutoff point
        cutoff = 4000
        cut_df = df[['num_tokens_spacy', 'num_tokens_bert']]
        cut_df.num_tokens_spacy = cut_df.num_tokens_spacy.clip(upper=cutoff)
        cut_df.num_tokens_bert = cut_df.num_tokens_bert.clip(upper=cutoff)

        hist_df = pd.concat([cut_df.num_tokens_spacy, cut_df.num_tokens_bert], keys=['spacy', 'bert']).to_frame()
        hist_df = hist_df.reset_index(level=0)
        hist_df = hist_df.rename(columns={'level_0': 'tokenizer', 0: 'Number of tokens'})

        plot = sns.displot(hist_df, x="Number of tokens", hue="tokenizer",
                           bins=100, kde=True, fill=True, height=5, aspect=2.5, legend=False)
        plot.set(xticks=list(range(0, 4500, 500)))
        plt.ylabel('Number of court cases')
        plt.legend(["BERT", "SpaCy"], loc='upper right', title='Tokenizer', fontsize=16, title_fontsize=18)
        plot.savefig(split_folder / f'{feature_col}_input_length_distribution-histogram.png', bbox_inches="tight")
        plt.clf()

        plot = sns.displot(hist_df, x="Number of tokens", hue="tokenizer", kind="ecdf", legend=False)
        plt.ylabel('Number of court cases')
        plt.legend(["BERT", "SPaCy"], loc='lower right', title='Tokenizer')
        plot.savefig(split_folder / f'{feature_col}_input_length_distribution-cumulative.png', bbox_inches="tight")
        plt.clf()

        plot = sns.displot(cut_df, x="num_tokens_spacy", y="num_tokens_bert")
        plot.savefig(split_folder / f'{feature_col}_input_length_distribution-bivariate.png', bbox_inches="tight")
        plt.clf()

    @staticmethod
    def save_labels(labels, folder):
        """
        Saves the labels and the corresponding ids as a json file
        :param labels:      the labels dict
        :param folder:   where to save the labels
        :return:
        """
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
            with open(f"{file_name}", 'w', encoding='utf-8') as f:
                json.dump(json_labels, f, ensure_ascii=False, indent=4)


    @staticmethod
    def split_date_stratified(df, split):
        """
        Splits the df into train, val and test based on the date
        :param df:      the df to be split
        :param split:   the exact split (how much of the data goes into train, val and test respectively)
        :return:
        """
        # TODO revise this for datasets including cantonal data and include year 2021

        train = df[df.year.isin(range(2000, 2016))]  # 16 Jahre
        val = df[df.year.isin(range(2016, 2018))]  # 2 Jahre
        test = df[df.year.isin(range(2018, 2021))]  # 3 Jahre
        secret_test = df[df.year.isin(range(2021, 2023))]  # 2 Jahre

        return train, val, test, secret_test

    def split_random(self, df, split):
        """
        Splits the df randomly into train, val and test
        :param df:      the df to be split
        :param split:   the exact split (how much of the data goes into train, val and test respectively)
        :return:
        """
        train, val, test = dd.from_pandas(df, npartitions=1).random_split(list(split), random_state=self.seed)

        # get pandas dfs again
        train = train.compute(scheduler='processes')
        val = val.compute(scheduler='processes')
        test = test.compute(scheduler='processes')

        return train, val, test

    @abc.abstractmethod
    def plot_custom(self, df, split_folder, folder):
        raise NotImplementedError("This method should be implemented in the subclass.")
