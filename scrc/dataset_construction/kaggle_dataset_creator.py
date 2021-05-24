import configparser
from collections import Counter
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from root import ROOT_DIR
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.log_utils import get_logger
import json

"""
Datasets to be created:
- Judgement prediction BGer:
    - text classification
    - input (considerations) to label (judgement)
- Citation prediction
    - multilabel text classification
    - input (entire text with citations removed) to label (cited laws and court decisions)
    - Features:
        - Zero/One/Few Shot
- Chamber/court prediction => proxy for legal area
    - text classification
    - input (situation) to label (chamber/court code)
    - Features:
        - Zero/One/Few Shot
- Date prediction
    - text classification
    - input (entire text) to label (date in different granularities: year, quarter, regression)
    - Features:
        - Zero/One/Few Shot
- Section splitting: 
    - token classification (text zoning task)
    - input (entire text) to label (section tags per token)
- LM Pretraining:
    - masked language modeling
    - input (entire text)
    - Features:
        - largest openly published corpus of court decisions

- level of appeal prediction?
    - ask Magda to annotate court_chambers.json with level of appeal for every court

Features:
- Time-stratified: train, val and test from different time ranges to simulate more realistic test scenario
- multilingual: 3 languages
- diachronic: more than 20 years
- very diverse: 3 languages, 26 cantons, 112 courts, 287 chambers, most if not all legal areas and all Swiss levels of appeal
"""


class DatasetCreator(DatasetConstructorComponent):
    """
    Creates the files for a kaggle dataset.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.seed = 42
        self.debug = False

    def create_datasets(self):
        engine = self.get_engine(self.db_scrc)

        datasets = ['judgement_prediction', 'citation_prediction', ]
        processed_file_path = self.data_dir / f"datasets_created.txt"
        datasets, message = self.compute_remaining_parts(processed_file_path, datasets)
        self.logger.info(message)

        for dataset in datasets:
            self.create_dataset(engine, dataset)
            self.mark_as_processed(processed_file_path, dataset)

    def create_dataset(self, engine, dataset):
        self.logger.info(f"Creating {dataset} dataset")

        create_df = getattr(self, dataset)  # get the function called by the dataset name

        folder = self.create_dir(self.datasets_subdir, dataset)
        split_type = "date-stratified"
        if dataset == "date_prediction":
            split_type = "random"  # if we determined the splits by date the labelset would be very small
        df, labels = create_df(engine)
        self.save_dataset(df, labels, folder, split_type)

    def judgement_prediction(self, engine):
        df = self.get_df(engine, 'considerations', 'judgements')
        df = df.rename(columns={"considerations": "text", "judgements": "label"})  # normalize column names
        labels, _ = list(np.unique(np.hstack(df.label), return_index=True))
        return df, labels

    def citation_prediction(self, engine):
        df = self.get_df(engine, 'text', 'citations')

        def replace_citations(series):
            series['label'] = []
            for law in series.citations['laws']:
                citation = law['text']
                series.text = series.text.replace(citation, "<law-citation>")
                series.label.append(citation)
            for ruling in series.citations['rulings']:
                citation = ruling['text']
                series.text = series.text.replace(citation, "<ruling-citation>")
                series.label.append(ruling['text'])

            return series

        df = df.apply(replace_citations, axis='columns')
        df = df.rename(columns={"text": "text"})  # normalize column names
        labels, _ = list(np.unique(np.hstack(df.label), return_index=True))
        return df, labels

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
        self.logger.info("Loading the data from the database")
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

        # save regular dataset
        self.logger.info("Saving the regular dataset")
        regular_dir = self.create_dir(folder, 'regular')
        train.to_csv(regular_dir / 'train.csv', index_label='id')
        val.to_csv(regular_dir / 'val.csv', index_label='id')
        test.to_csv(regular_dir / 'test.csv', index_label='id')
        self.save_labels(labels, regular_dir / 'labels.json')

        self.save_report(df, folder)

        if kaggle:
            self.save_kaggle_dataset(folder, labels, test, train, val)

        self.logger.info(f"Saved dataset files to {folder}")

    def save_report(self, df, folder):
        self.logger.info("Computing metadata reports on the input lengths and the labels")

        # compute label imbalance
        ax = df.label.astype(str).hist()
        ax.tick_params(labelrotation=90)
        ax.get_figure().savefig(folder / 'multi_label_distribution.png', bbox_inches="tight")

        counter_dict = dict(Counter(np.hstack(df.label)))
        label_counts = pd.DataFrame.from_dict(counter_dict, orient='index', columns=['num_occurrences'])
        label_counts.to_csv(folder / 'single_label_distribution.csv', index_label='label')

        ax = label_counts.plot.bar(y='num_occurrences', rot=15)
        ax.get_figure().savefig(folder / 'single_label_distribution.png')

        # compute median input length
        df.num_tokens.describe().to_csv(folder / 'input_length_distribution.csv', index_label='measure')

    def save_kaggle_dataset(self, folder, labels, test, train, val):
        self.logger.info("Saving the data in kaggle format")
        # create solution file
        solution = test.drop('text', axis='columns')  # drop text
        solution = solution.rename(columns={"label": "Expected"})  # rename according to kaggle conventions
        # create test file
        test = test.drop('label', axis='columns')  # drop label
        # create sampleSubmission file
        sample_submission = solution.rename(columns={"Expected": "Predicted"})  # rename according to kaggle conventions
        sample_submission['Predicted'] = np.random.choice(labels, size=len(solution))  # set to random value

        # save special kaggle files
        kaggle_dir = self.create_dir(folder, 'kaggle')
        train.to_csv(kaggle_dir / 'train.csv', index_label='Id')
        val.to_csv(kaggle_dir / 'val.csv', index_label='Id')
        test.to_csv(kaggle_dir / 'test.csv', index_label='Id')
        solution.to_csv(kaggle_dir / 'solution.csv', index_label='Id')
        sample_submission.to_csv(kaggle_dir / 'sampleSubmission.csv', index_label='Id')

    def save_labels(self, labels, file_name):
        labels_dict = dict(enumerate(labels))
        json_labels = {"id2label": labels_dict, "label2id": {y: x for x, y in labels_dict.items()}}
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(json_labels, f, ensure_ascii=False, indent=4)

    def split_date_stratified(self, df, split):
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
        # split into train, val and test sets
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
