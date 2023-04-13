import os

import datasets
import json
from multiprocessing import Pool
import gc

import jsonlines
from tqdm import tqdm
from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.enums.section import Section
from scrc.utils.log_utils import get_logger
import numpy as np
import pandas as pd
import ast
from pandarallel import pandarallel
from scrc.utils.main_utils import get_config
from datasets import load_dataset, concatenate_datasets, interleave_datasets
import random


class PreprocessDoc2doc(DatasetCreator):
    """
    corpus = {
        "law_id_1": {
            "title": "sr_number",
            "text": "blabliblu“
        },
        "decision_id_bge_1": {
            "title": "sr_number“,
            "text": "blablebli“
    },
    }

    queries = {
        "decision_id_1": "facts or considerations",
        "decision_id_2": "facts or considerations"
    }

    qrels = {
        "decision_id_1": {"law_id_1": 1},
        "decision_id_2": {"decision_id_bge_1": 1},
    }
    important all ids used in qrels must be found in corpus
    """

    def __init__(self, config: dict, debug: bool = False):
        super().__init__(config, debug)
        self.logger = get_logger(__name__)
        self.split_type = "date-stratified"
        self.dataset_name = "doc2doc_ir"
        self.feature_cols = [Section.FACTS, Section.CONSIDERATIONS]
        self.reports_folder = self.create_dir(self.get_dataset_folder(), f'reports')
        pandarallel.initialize(progress_bar=True)
        tqdm.pandas()
        self.id_list = []
        self.feature_cols = ['facts', 'considerations']
        self.id_list = []
        self.read_from_files = True

    def save_to_json(self, dataset, path):
        dataset.to_json(path, orient='records', lines=True, force_ascii=False)

    def load_dataset(self):
        # load laws from huggingface
        dataset_train = load_dataset('rcds/doc2doc', split='train')
        dataset_validation = load_dataset('rcds/doc2doc', split='validation')
        dataset_test = load_dataset('rcds/doc2doc', split='test')
        dataset = interleave_datasets([dataset_train, dataset_validation, dataset_test])
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.filter(lambda example: not example["cited_rulings"].startswith("[]"))
        de_dataset = dataset.filter(lambda example: example['language'] == 'de')
        fr_dataset = dataset.filter(lambda example: example['language'] == 'fr')
        it_dataset = dataset.filter(lambda example: example['language'] == 'it')

        # dataset = dataset.select(list(range(0, 100))) # use this if you want to create a debug dataset

        columns_to_remove = [item for item in dataset.column_names if
                             item not in ['decision_id', 'facts', 'considerations', 'laws', 'cited_rulings']]
        it_dataset = it_dataset.remove_columns(columns_to_remove)
        fr_dataset = fr_dataset.remove_columns(columns_to_remove)
        de_dataset = de_dataset.remove_columns(columns_to_remove)
        dataset = dataset.remove_columns(columns_to_remove)

        return dataset, it_dataset, fr_dataset, de_dataset

    def clean_dataset(self, df):
        def create_list_citations(row):
            id_list = ast.literal_eval(row['cited_rulings']) + ast.literal_eval(row['laws'])
            id_list = list(set(id_list))
            return id_list

        df['citations'] = df.apply(create_list_citations, axis='columns')
        df['citations'] = df.citations.apply(lambda x: None if x == [] else x)
        df.drop(columns=['cited_rulings', 'laws'], inplace=True)
        df = df.dropna()
        return df

    def create_corpus(self):
        # process laws
        law_dataset = load_dataset('rcds/swiss_legislation', split='train')
        law_dataset = law_dataset.filter(lambda row: row['canton'] == 'ch')
        cols_to_remove = [item for item in law_dataset.column_names if
                          item not in ['uuid', 'sr_number', 'pdf_content', 'language']]
        law_dataset = law_dataset.remove_columns(cols_to_remove)
        law_dataset = law_dataset.rename_column("uuid", "id")
        law_dataset = law_dataset.rename_column("sr_number", "title")
        law_dataset = law_dataset.rename_column("pdf_content", "text")

        law_df = pd.DataFrame(law_dataset)
        law_df['id'] = law_df.id.astype(str)

        corpus_dict = dict()

        def write_dict(row):
            id = str(row['id'])
            corpus_dict[id] = {"title": row['title'], "text": row['text'], 'language': row['language']}
            return row

        law_df.apply(write_dict, axis="columns")

        # process rulings
        rulings_df = self.load_rulings()

        def write_rulings_dict(row):
            id = str(row['decision_id'])
            corpus_dict[id] = {"title": row['file_number'], "text": f"{row['facts']} {row['considerations']}", 'language': row['language']}
            return row

        rulings_df = rulings_df.apply(write_rulings_dict, axis='columns')

        rulings_df['text'] = rulings_df['facts'].astype(str) + ' ' + rulings_df['considerations'] .astype(str)
        cols_to_remove = [item for item in rulings_df.columns if
                          item not in ['decision_id', 'text', 'language', 'file_number']]
        rulings_df = rulings_df.drop(columns=cols_to_remove)
        rulings_df = rulings_df.rename(columns={"decision_id": "id", "file_number": 'title'})

        df = pd.concat([rulings_df, law_df])
        self.save_to_json(df, f"{self.get_dataset_folder()}/corpus_all.jsonl")

        return corpus_dict

    def load_rulings(self):
        ruling_dataset = load_dataset('rcds/swiss_rulings', split='train')
        decision_df = pd.DataFrame(ruling_dataset)
        print(f"BGE: There are {len(decision_df.index)} in db (also old or not referenced included).")
        return decision_df

    def create_qrels(self, df, corpus, language):
        """
                qrels = {
                "decision_id_1": {"law_id_1": 1},
                "decision_id_2": {"decision_id_bge_1F": 1},
                    }
                    """
        df = df.explode('citations')
        df = df.dropna()
        self.counter = 0

        def filter_cits(cit):
            try:
                match = corpus[cit]
                if match['language'] == language or language == 'mixed':
                    self.counter += 1
                    return True
            except Exception as e:
                pass
            return None

        df['match'] = df.citations.apply(filter_cits)
        self.logger.info(f"there are {self.counter} citations with valid ids")

        df = df.dropna()

        qrels_dict = {}
        corpus_dict = {}
        queries_dict = {}

        def write_qrels_dict(row):
            id = row['decision_id']
            corpus_id = row['citations']
            if id not in qrels_dict:
                qrels_dict[id] = {corpus_id: 1}
                queries_dict[id] = row['facts']
            else:
                qrels_dict[id][corpus_id] = 1
            if corpus_id not in corpus_dict:
                # corpus_df.loc[len(df.index)] = ['id', 'text', 'language', 'title']
                corpus_dict[corpus_id] = str(corpus[corpus_id]['text'])
            return row

        qrels_df = df.apply(write_qrels_dict, axis="columns")
        qrels_df = qrels_df.rename(columns={'decision_id': 'id', 'citations': 'corp_id'})
        qrels_df = qrels_df.drop(columns=['facts', 'considerations'])
        print(f"Save to: {self.get_dataset_folder()}/<name>_{language}.jsonl")
        self.save_to_json(qrels_df, f"{self.get_dataset_folder()}/qrels_{language}.jsonl")

        corpus_df = pd.DataFrame(corpus_dict.items(), columns=['id', 'text'])
        self.save_to_json(corpus_df, f"{self.get_dataset_folder()}/corpus_{language}.jsonl")

        queries_df = pd.DataFrame(queries_dict.items(), columns=['id', 'text'])
        self.save_to_json(queries_df, f"{self.get_dataset_folder()}/queries_{language}.jsonl")

        queries_df['language'] = language
        qrels_df['language'] = language
        corpus_df['language'] = language

        return queries_df, qrels_df, corpus_df

    def create_triplets(self, df, feature, corpus):
        """
        triplets:
        1. bger_text (facts or considerations)
        2. cited ruling or law text
        3. ruling or law which was NOT cited text
        """

        # create sample triplets:
        # 1. pick random 100 bger samples -> already done by providing df
        df['cit_list'] = df['citations']

        # 2. go explode bger for each cited law and ruling
        df = df.explode('citations')
        print(df.citations.tolist())

        # 3. find for each law / ruling the text
        self.counter = 0
        def write_text(row):
            try:
                text = corpus[row['citations']]
                return text
            except Exception as e:
                pass
            self.counter += 1
            return None

        df['citations'] = df.apply(write_text, axis='columns')
        self.logger.info(f"There are {self.counter} invalid cit ids.")
        # 4. find a good example of law / ruling not in citaions of that case
        def find_neg_example(row):
            not_found = True
            while not_found:
                corp_id, corp_text = random.choice(list(corpus.items()))
                if corp_id not in row['cit_list']:
                    not_found = False
            return corp_text['text']

        df['neg_text'] = df.apply(find_neg_example, axis='columns')

        # 5. get the right format  List[Tuple[str, str, str]]
        columns_to_remove = [item for item in df.columns if item not in [feature, 'citations', 'neg_text']]
        df = df.drop(columns=columns_to_remove)
        df = df.dropna()

        triples = []

        def write_list(row):
            triples.append((row[feature], row['citations'], row['neg_text']))
            return row

        df = df.apply(write_list, axis='columns')
        self.save_to_json(df, f"{self.get_dataset_folder()}/triplets.jsonl")
        return triples


if __name__ == '__main__':
    config = get_config()
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)
    preprocessDoc2doc = PreprocessDoc2doc(config)
    full_corpus = preprocessDoc2doc.create_corpus()
    dataset, it_dataset, fr_dataset, de_dataset = preprocessDoc2doc.load_dataset()

    df = pd.DataFrame(dataset)
    df = preprocessDoc2doc.clean_dataset(df)
    queries_df, qrels_df, corpus_df = preprocessDoc2doc.create_qrels(df, full_corpus, 'mixed')

    it_df = pd.DataFrame(it_dataset)
    it_df = preprocessDoc2doc.clean_dataset(it_df)
    queries_df_it, qrels_df_it, corpus_df_it = preprocessDoc2doc.create_qrels(it_df, full_corpus, 'it')

    fr_df = pd.DataFrame(fr_dataset)
    fr_df = preprocessDoc2doc.clean_dataset(fr_df)
    queries_df_fr, qrels_df_fr, corpus_df_fr = preprocessDoc2doc.create_qrels(fr_df, full_corpus, 'fr')

    de_df = pd.DataFrame(de_dataset)
    de_df = preprocessDoc2doc.clean_dataset(de_df)
    queries_df_de, qrels_df_de, corpus_df_de = preprocessDoc2doc.create_qrels(de_df, full_corpus, 'de')

    df = pd.concat([queries_df, queries_df_de, queries_df_fr, queries_df_it])
    preprocessDoc2doc.save_to_json(df, f"{preprocessDoc2doc.get_dataset_folder()}/queries_all.jsonl")

    df = pd.concat([qrels_df, qrels_df_de, qrels_df_fr, qrels_df_it])
    preprocessDoc2doc.save_to_json(df, f"{preprocessDoc2doc.get_dataset_folder()}/qrels_all.jsonl")

    df = pd.concat([corpus_df, corpus_df_it, corpus_df_fr, corpus_df_de])
    preprocessDoc2doc.save_to_json(df, f"{preprocessDoc2doc.get_dataset_folder()}/corpus_all.jsonl")

    # triplets = preprocessDoc2doc.create_triplets(df, 'facts', full_corpus)
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

