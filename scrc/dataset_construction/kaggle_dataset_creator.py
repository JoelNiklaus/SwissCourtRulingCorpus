"""
Take cleaned German aggregation
keep columns chamber and text
sample 100 decisions from each of the 10 most frequent chambers
shuffle
split into train, validation and test set
remove labels
possible labels: chamber, (date)

"""
import configparser
import dask.dataframe as dd
import numpy as np
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


class KaggleDatasetCreator(DatasetConstructorComponent):
    """
    Creates the files for a kaggle dataset.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.seed = 42

    def create_datasets(self):
        engine = self.get_engine(self.db_scrc)

        datasets = ['judgement_prediction', 'citation_prediction', ]
        # 'chamber_prediction', 'date_prediction','section_splitting']
        processed_file_path = self.data_dir / f"datasets_created.txt"
        datasets, message = self.compute_remaining_parts(processed_file_path, datasets)
        self.logger.info(message)

        for dataset in datasets:
            self.create_dataset(engine, dataset)

    def create_dataset(self, engine, dataset):
        self.logger.info(f"Creating {dataset} Dataset")

        create_df = getattr(self, dataset)  # get the function called by the dataset name

        folder = self.create_dir(self.kaggle_subdir, dataset)
        self.create_kaggle_dataset(create_df(engine), folder)

    def judgement_prediction(self, engine):
        columns = "extract(year from date) as year, considerations, judgement"
        where = "court = 'CH_BGer' AND considerations IS NOT NULL AND judgement IS NOT NULL"
        order_by = "year"
        df = next(self.select(engine, 'de', columns=columns, where=where, order_by=order_by, chunksize=int(1e7)))

        df = df.rename(columns={"considerations": "text", "judgement": "label"})  # normalize column names
        return df

    def citation_prediction(self, engine):
        columns = "extract(year from date) as year, text, citations"
        where = "court = 'CH_BGer' AND text IS NOT NULL AND citations IS NOT NULL"
        order_by = "year"
        df = next(self.select(engine, 'de', columns=columns, where=where, order_by=order_by, chunksize=int(1e7)))

        def replace_citations(series):
            laws = series.citations['laws']
            rulings = series.citations['rulings']

            for law in laws:
                series.text = series.text.replace(law['text'], "<law-citation>")
            for ruling in rulings:
                series.text = series.text.replace(ruling['text'], "<ruling-citation>")

        df = df.apply(replace_citations, axis='columns')

        df = df.rename(columns={"text": "text", "citations": "label"})  # normalize column names

        return df

    def chamber_prediction(self, engine):
        pass

    def date_prediction(self, engine):
        pass

    def section_splitting(self, engine):
        pass

    def create_kaggle_dataset(self, df, folder, split_type="date-stratified"):
        """
        creates all the files necessary for a kaggle dataset from a given df
        :param df:          needs to contain the columns text and label
        :param folder:      where to save the files
        :param split_type:  "date-stratified" or "random"
        :return:
        """
        self.logger.info("Splitting data into train, val and test set")
        if split_type == "random":
            train, val, test = self.split_random(df)
        elif split_type == "date-stratified":
            train, val, test = self.split_date_stratified(df)
        else:
            raise ValueError("Please supply a valid split_type")

        # create solution file
        solution = test.drop('text', axis='columns')  # drop text
        solution = test.rename(columns={"label": "Expected"})  # rename according to kaggle conventions

        # create test file
        test = test.drop('label', axis='columns')  # drop label

        # create sampleSubmission file
        labels = df.label.unique()
        sample_submission = solution.rename(columns={"Expected": "Predicted"})  # rename according to kaggle conventions
        sample_submission['Predicted'] = np.random.choice(list(labels), size=len(solution))  # set to random value

        # save to folder
        train.to_csv(folder / 'train.csv', index_label='Id')
        val.to_csv(folder / 'val.csv', index_label='Id')
        test.to_csv(folder / 'test.csv', index_label='Id')
        solution.to_csv(folder / 'solution.csv', index_label='Id')
        sample_submission.to_csv(folder / 'sampleSubmission.csv', index_label='Id')
        self.save_labels(labels, folder / 'labels.json')
        self.logger.info(f"Saved files necessary for competition to {folder}")

    def save_labels(self, labels, file_name):
        labels_dict = dict(enumerate(labels))
        json_labels = {"idx2label": labels_dict, "label2idx": {y: x for x, y in labels_dict.items()}}
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(json_labels, f, ensure_ascii=False, indent=4)

    def split_date_stratified(self, df, split=(0.7, 0.1, 0.2)):
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

    def split_random(self, df, split=(0.7, 0.1, 0.2)):
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

    kaggle_dataset_creator = KaggleDatasetCreator(config)
    kaggle_dataset_creator.create_datasets()
