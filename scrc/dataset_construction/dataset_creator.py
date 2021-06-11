import configparser
import inspect
from collections import Counter
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from root import ROOT_DIR
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.log_utils import get_logger
import json

from scrc.utils.main_utils import string_contains_one_of_list
from scrc.utils.term_definitions_extractor import TermDefinitionsExtractor

"""
TODO: Find justification (Why?) for these tasks: talk this through with Matthias/Ilias

Datasets to be created:
- Judgement prediction BGer:
    - text classification
    - input (considerations/situation) to label (judgement)
    - Why?
- Citation prediction
    - multilabel text classification
    - input (entire text with citations removed) to label (cited laws and court decisions)
    - Features:
        - Zero/One/Few Shot
    - Why?: 
- Citation Type labeling:
    - fill mask
    - input (entire text with citations removed by mask token)
    - Tasks: predict citation type (law or ruling) or citation
    - Why?: difficult LM Pretraining task
- Chamber/court prediction
    - text classification
    - input (situation) to label (chamber/court code)
    - Features:
        - Zero/One/Few Shot
    - Why?: proxy for legal area prediction to help lawyers find suitable laws
- Date prediction
    - text classification
    - input (entire text) to label (date in different granularities: year, quarter, regression)
    - Features:
        - Zero/One/Few Shot
    - Why?: learn temporal data shift
- Section splitting: 
    - token classification (text zoning task)
    - input (entire text) to label (section tags per token)
    - Why?: 
- LM Pretraining:
    - masked language modeling
    - input (entire text)
    - Features:
        - largest openly published corpus of court decisions
    - Why?: train Swiss Legal BERT

- level of appeal prediction?
    - TODO ask Magda to annotate court_chambers.json with level of appeal for every court

Features:
- Time-stratified: train, val and test from different time ranges to simulate more realistic test scenario
- multilingual: 3 languages
- diachronic: more than 20 years
- very diverse: 3 languages, 26 cantons, 112 courts, 287 chambers, most if not all legal areas and all Swiss levels of appeal
"""


class DatasetCreator(DatasetConstructorComponent):
    """
    Retrieves the data and preprocesses it for subdatasets of SCRC.
    Also creates the necessary files for a kaggle dataset.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.seed = 42
        self.debug = True
        self.num_ruling_citations = 1000  # the 1000 most common ruling citations will be included

    def create_datasets(self):
        engine = self.get_engine(self.db_scrc)

        datasets = ['facts_judgement_prediction', 'considerations_judgement_prediction', 'citation_prediction']
        processed_file_path = self.data_dir / f"datasets_created.txt"
        datasets, message = self.compute_remaining_parts(processed_file_path, datasets)
        self.logger.info(message)

        for dataset in datasets:
            self.create_dataset(engine, dataset)
            self.mark_as_processed(processed_file_path, dataset)

    def create_dataset(self, engine, dataset):
        """
        Retrieves the respective function named by the dataset and executes it to get the df for that dataset.

        :param engine:  the engine to retrieve the data
        :param dataset: the name of the dataset
        :return:
        """
        self.logger.info(f"Creating {dataset} dataset")

        create_df = getattr(self, dataset)  # get the function called by the dataset name

        folder = self.create_dir(self.datasets_subdir, dataset)
        split_type = "date-stratified"
        if dataset == "date_prediction":
            split_type = "random"  # if we determined the splits by date the labelset would be very small
        df, labels = create_df(engine)
        self.save_dataset(df, labels, folder, split_type)

    def facts_judgement_prediction(self, engine):
        return self.judgement_prediction(engine, 'facts')

    def considerations_judgement_prediction(self, engine):
        return self.judgement_prediction(engine, 'considerations')

    def judgement_prediction(self, engine, input, with_write_off=False, with_partials=False, make_single_label=True):
        df = self.get_df(engine, input, 'judgements')

        def clean(judgements):
            out = []
            if not with_write_off:
                # remove write_off because reason for it happens mostly behind the scenes and not written in the facts
                if 'write_off' in judgements:
                    return np.nan  # set to nan so it can later be dropped easily

            for judgement in judgements:
                # remove "partial_" from all the items to merge them with full ones
                if not with_partials:
                    judgement = judgement.replace("partial_", "")

                out.append(judgement)

            # remove all labels which are complex combinations (reason: different questions => model cannot know which one to pick)
            if make_single_label:
                # contrary judgements point to multiple questions which is too complicated
                if 'dismissal' in out and 'approval' in out:
                    return np.nan
                # if we have inadmissible and another one, we just remove inadmissible
                if 'inadmissible' in out and len(out) > 1:
                    out.remove('inadmissible')
                if len(out) > 1:
                    message = f"By now we should only have one label. But instead we still have the labels {out}"
                    raise ValueError(message)
                return out[0]  # just return the first label because we only have one left

            return out

        df.judgements = df.judgements.apply(clean)
        df = df.dropna()  # drop empty labels introduced by cleaning before

        df = df.rename(columns={input: "text", "judgements": "label"})  # normalize column names
        labels, _ = list(np.unique(np.hstack(df.label), return_index=True))
        return df, labels

    def citation_prediction(self, engine):
        df = self.get_df(engine, 'text', 'citations')

        this_function_name = inspect.currentframe().f_code.co_name
        folder = self.create_dir(self.datasets_subdir, this_function_name)

        # calculate most common BGE citations
        most_common_rulings = self.get_most_common_citations(df, folder, 'rulings')

        # list with only the most common laws
        most_common_laws = self.get_most_common_citations(df, folder, 'laws')

        laws_from_term_definitions = self.get_laws_from_term_definitions()

        # TODO extend laws to other languages as well.
        #  IMPORTANT: you need to take care of the fact that the laws are named differently in each language but refer to the same law!

        def replace_citations(series, ref_mask_token="<ref>"):
            # TODO think about splitting laws and rulings into two separate labels
            labels = set()
            for law in series.citations['laws']:
                citation = law['text']
                found_string_in_list = string_contains_one_of_list(citation, laws_from_term_definitions['de'])
                if found_string_in_list:
                    series.text = series.text.replace(citation, ref_mask_token)
                    labels.add(found_string_in_list)
            for ruling in series.citations['rulings']:
                citation = ruling['text']
                if string_contains_one_of_list(citation, most_common_rulings):
                    series.text = series.text.replace(citation, ref_mask_token)
                    labels.add(citation)
            series['label'] = list(labels)
            return series

        df = df.apply(replace_citations, axis='columns')
        df = df.rename(columns={"text": "text"})  # normalize column names
        labels, _ = list(np.unique(np.hstack(df.label), return_index=True))
        return df, labels

    def get_most_common_citations(self, df, folder, type, plot_n_most_common=10):
        """
        Retrieves the most common citations of a given type (rulings/laws).
        Additionally plots the plot_n_most_common most common citations
        :param df:
        :param folder:
        :param type:
        :return:
        """
        valid_types = ['rulings', 'laws']
        if type not in valid_types:
            raise ValueError(f"Please supply a valid citation type from {valid_types}")
        type_citations = []
        for citations in df.citations:
            for type_citation in citations[type]:
                type_citations.append(type_citation['text'])
        most_common_with_frequency = Counter(type_citations).most_common(self.num_ruling_citations)
        print(most_common_with_frequency)

        # Plot the 10 most common citations
        # remove BGG articles because they are obvious
        most_common_interesting = [(k, v) for k, v in most_common_with_frequency if 'BGG' not in k]
        ruling_citations = pd.DataFrame.from_records(most_common_interesting[:plot_n_most_common],
                                                     columns=['citation', 'frequency'])
        ax = ruling_citations.plot.bar(x='citation', y='frequency', rot=90)
        ax.get_figure().savefig(folder / f'most_common_{type}_citations.png', bbox_inches="tight")

        return list(dict(most_common_with_frequency).keys())

    def get_laws_from_term_definitions(self):
        term_definitions_extractor = TermDefinitionsExtractor()
        term_definitions = term_definitions_extractor.extract_term_definitions()
        laws_from_term_definitions = {lang: [] for lang in self.languages}
        for lang in self.languages:
            for definition in term_definitions[lang]:
                for synonyms in definition:
                    if synonyms['type'] == 'ab':  # ab stands for abbreviation
                        # append the string of the abbreviation
                        laws_from_term_definitions[lang].append(synonyms['text'])
        return laws_from_term_definitions

    def chamber_prediction(self, engine):
        # TODO process in batches because otherwise too large
        df = self.get_df(engine, 'situation', 'chamber')
        df = df.rename(columns={"situation": "text", "chamber": "label"})  # normalize column names
        labels = list(df.label.unique())
        return df, labels

    def date_prediction(self, engine):
        # TODO process in batches because otherwise too large
        df = self.get_df(engine, 'text', 'date')
        df = df.rename(columns={"situation": "text", "date": "label"})  # normalize column names
        labels = list(df.label.unique())
        return df, labels

    def section_splitting(self, engine):
        pass

    def get_df(self, engine, feature_col, label_col):
        self.logger.info("Started loading the data from the database")
        columns = f"extract(year from date) as year, num_tokens, {feature_col}, {label_col}"
        where = f"{feature_col} IS NOT NULL AND {feature_col} != '' AND {label_col} IS NOT NULL"
        order_by = "year"
        if self.debug:
            chunksize = 2e2  # run on smaller dataset for testing
        else:
            chunksize = 2e5
        df = next(self.select(engine, 'de', columns=columns, where=where, order_by=order_by, chunksize=int(chunksize)))
        df.year = df.year.astype(int)
        df = self.clean_from_empty_strings(df, feature_col)
        self.logger.info("Finished loading the data from the database")
        return df

    def clean_from_empty_strings(self, df, column):
        # replace empty and whitespace strings with nan so that they can be removed
        df[column] = df[column].replace(r'^\s*$', np.nan, regex=True)
        df = df.dropna()  # drop null values not recognized by sql where clause
        df = df.reset_index(drop=True)  # reindex to get nice indices
        return df

    def save_dataset(self, df: pd.DataFrame, labels: list, folder: Path,
                     split_type="date-stratified", split=(0.7, 0.1, 0.2), kaggle=True):
        """
        creates all the files necessary for a kaggle dataset from a given df
        :param df:          needs to contain the columns text and label
        :param labels:      all the labels
        :param folder:      where to save the files
        :param split_type:  "date-stratified" or "random"
        :param split:       how to split the data into train, val and test set: needs to sum up to 1
        :param kaggle:      whether or not to create the special kaggle dataset
        :return:
        """
        self.logger.info("Splitting data into train, val and test set")
        if split_type == "random":
            train, val, test = self.split_random(df, split)
        elif split_type == "date-stratified":
            train, val, test = self.split_date_stratified(df, split)
        else:
            raise ValueError("Please supply a valid split_type")

        self.save_report(df, folder)

        # save regular dataset
        self.logger.info("Saving the regular dataset")
        regular_dir = self.create_dir(folder, 'regular')
        train.to_csv(regular_dir / 'train.csv', index_label='id')
        val.to_csv(regular_dir / 'val.csv', index_label='id')
        test.to_csv(regular_dir / 'test.csv', index_label='id')
        self.save_labels(labels, regular_dir / 'labels.json')

        if kaggle:
            self.save_kaggle_dataset(folder, train, val, test)

        self.logger.info(f"Saved dataset files to {folder}")

    def save_report(self, df, folder):
        """
        Saves statistics about the dataset in the form of csv tables and png graphs.
        :param df:      the df containing the dataset
        :param folder:  the base folder to save the report to
        :return:
        """
        self.logger.info("Computing metadata reports on the input lengths and the labels")

        # compute label imbalance
        ax = df.label.astype(str).hist()
        ax.tick_params(labelrotation=90)
        ax.get_figure().savefig(folder / 'multi_label_distribution.png', bbox_inches="tight")

        counter_dict = dict(Counter(np.hstack(df.label)))
        label_counts = pd.DataFrame.from_dict(counter_dict, orient='index', columns=['num_occurrences'])
        label_counts.to_csv(folder / 'single_label_distribution.csv', index_label='label')

        ax = label_counts.plot.bar(y='num_occurrences', rot=15)
        ax.get_figure().savefig(folder / 'single_label_distribution.png', bbox_inches="tight")

        # compute median input length
        df.num_tokens.describe().to_csv(folder / 'input_length_distribution.csv', index_label='measure')

    def save_kaggle_dataset(self, folder, train, val, test):
        """
        Saves the train, val and test set to a kaggle-style dataset
        and also creates a solution and sampleSubmission file.
        :param folder:  where to save the dataset
        :param train:   the train set
        :param val:     the val set
        :param test:    the test set
        :return:
        """
        self.logger.info("Saving the data in kaggle format")
        # create solution file
        solution = test.drop('text', axis='columns')  # drop text
        solution = solution.rename(columns={"label": "Expected"})  # rename according to kaggle conventions
        # create test file
        test = test.drop('label', axis='columns')  # drop label
        # create sampleSubmission file
        sample_submission = solution.rename(columns={"Expected": "Predicted"})  # rename according to kaggle conventions
        sample_submission['Predicted'] = np.random.choice(solution.Expected, size=len(solution))  # set to random value

        # save special kaggle files
        kaggle_dir = self.create_dir(folder, 'kaggle')
        train.to_csv(kaggle_dir / 'train.csv', index_label='Id')
        val.to_csv(kaggle_dir / 'val.csv', index_label='Id')
        test.to_csv(kaggle_dir / 'test.csv', index_label='Id')
        solution.to_csv(kaggle_dir / 'solution.csv', index_label='Id')
        sample_submission.to_csv(kaggle_dir / 'sampleSubmission.csv', index_label='Id')

    def save_labels(self, labels, file_name):
        """
        Saves the labels and the corresponding ids as a json file
        :param labels:      the labels dict
        :param file_name:   where to save the labels
        :return:
        """
        labels_dict = dict(enumerate(labels))
        json_labels = {"id2label": labels_dict, "label2id": {y: x for x, y in labels_dict.items()}}
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(json_labels, f, ensure_ascii=False, indent=4)

    def split_date_stratified(self, df, split):
        """
        Splits the df into train, val and test based on the date
        :param df:      the df to be split
        :param split:   the exact split (how much of the data goes into train, val and test respectively)
        :return:
        """
        last_year = 2020  # disregard partial year 2021
        first_year = 2000  # before the data is quite sparse and there might be too much differences in the language
        num_years = last_year - first_year + 1

        test_years = int(split[2] * num_years)
        val_years = int(split[1] * num_years)

        test_start_year = last_year - test_years + 1
        val_start_year = test_start_year - val_years

        test_range = range(test_start_year, last_year + 1)  # 2000 - 2014
        val_range = range(val_start_year, test_start_year)  # 2015 - 2016
        train_range = range(first_year, val_start_year)  # 2017 - 2020

        test = df[df.year.isin(test_range)]
        val = df[df.year.isin(val_range)]
        train = df[df.year.isin(train_range)]

        return train, val, test

    def split_random(self, df, split):
        """
        Splits the df randomly into train, val and test
        :param df:
        :param split:
        :return:
        """
        train, val, test = dd.from_pandas(df, npartitions=1).random_split(list(split), random_state=self.seed)

        # get pandas dfs again
        train = train.compute(scheduler='processes')
        val = val.compute(scheduler='processes')
        test = test.compute(scheduler='processes')

        return train, val, test


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    dataset_creator = DatasetCreator(config)
    dataset_creator.create_datasets()
