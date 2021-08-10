import configparser
import inspect
from collections import Counter
from pathlib import Path

import seaborn as sns

import dask.dataframe as dd
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from root import ROOT_DIR
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.log_utils import get_logger
import json

from spacy.lang.de import German
from spacy.lang.fr import French
from spacy.lang.it import Italian

from scrc.utils.main_utils import string_contains_one_of_list, get_legal_area, legal_areas, get_region
from scrc.utils.term_definitions_extractor import TermDefinitionsExtractor

# pd.options.mode.chained_assignment = None  # default='warn'

"""
Extend datasets with big cantonal courts? => only if it does not take too much time (1-2 day per court)
Datasets to be created:
- Judgements
    - Judgement prediction BGer:
        - text classification
        - input (considerations/facts) to label (judgement)
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


class DatasetCreator(DatasetConstructorComponent):
    """
    Retrieves the data and preprocesses it for subdatasets of SCRC.
    Also creates the necessary files for a kaggle dataset.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.seed = 42
        self.debug = False
        self.debug_chunksize = 2e2
        self.num_ruling_citations = 1000  # the 1000 most common ruling citations will be included

    def create_datasets(self):
        engine = self.get_engine(self.db_scrc)

        datasets = ['facts_judgement_prediction']  # , 'considerations_judgement_prediction', 'citation_prediction']
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

        split_type = "date-stratified"
        if dataset == "date_prediction":
            split_type = "random"  # if we determined the splits by date the labelset would be very small

        dataset_folder = self.create_dir(self.datasets_subdir, dataset)
        for lang in self.languages:
            self.logger.info(f"Processing language {lang}")
            df, labels = getattr(self, dataset)(engine, lang)  # get and call the function called by the dataset name
            lang_folder = self.create_dir(dataset_folder, lang)
            self.save_dataset(df, labels, lang_folder, split_type)

    def facts_judgement_prediction(self, engine, lang):
        return self.judgement_prediction(engine, 'facts', lang)

    def considerations_judgement_prediction(self, engine, lang):
        return self.judgement_prediction(engine, 'considerations', lang)

    def judgement_prediction(self, engine, input, lang, with_write_off=False, with_unification=False,
                             with_inadmissible=False, with_partials=False, make_single_label=True):
        df = self.get_df(engine, input, 'judgements', lang)

        # Delete cases with "Nach Einsicht" from the dataset because they are mostly inadmissible or otherwise dismissal
        # => too easily learnable for the model (because of spurious correlation)
        if with_inadmissible:
            df = df[~df[input].str.startswith('Nach Einsicht')]

        def clean(judgements):
            out = set()
            for judgement in judgements:
                # remove "partial_" from all the items to merge them with full ones
                if not with_partials:
                    judgement = judgement.replace("partial_", "")

                out.add(judgement)

            if not with_write_off:
                # remove write_off because reason for it happens mostly behind the scenes and not written in the facts
                if 'write_off' in judgements:
                    out.remove('write_off')

            if not with_unification:
                # remove unification because reason for it happens mostly behind the scenes and not written in the facts
                if 'unification' in judgements:
                    out.remove('unification')

            if not with_inadmissible:
                # remove inadmissible because it is a formal reason and not that interesting semantically.
                # Facts are formulated/summarized in a way to justify the decision of inadmissibility
                # hard to know solely because of the facts (formal reasons, not visible from the facts)
                if 'inadmissible' in judgements:
                    out.remove('inadmissible')

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
                elif len(out) == 1:
                    return out.pop()  # just return the first label because we only have one left
                else:
                    return np.nan

            return list(out)

        df = df.dropna(subset=['judgements'])
        df.judgements = df.judgements.apply(clean)
        df = df.dropna(subset=['judgements'])  # drop empty labels introduced by cleaning before

        df = df.rename(columns={input: "text", "judgements": "label"})  # normalize column names
        labels, _ = list(np.unique(np.hstack(df.label), return_index=True))
        return df, labels

    def citation_prediction(self, engine, lang):
        df = self.get_df(engine, 'text', 'citations', lang)

        this_function_name = inspect.currentframe().f_code.co_name
        folder = self.create_dir(self.datasets_subdir, this_function_name)

        # calculate most common BGE citations
        most_common_rulings = self.get_most_common_citations(df, folder, 'rulings')

        # list with only the most common laws
        most_common_laws = self.get_most_common_citations(df, folder, 'laws')

        law_abbr_by_lang = self.get_law_abbr_by_lang()

        #  IMPORTANT: we need to take care of the fact that the laws are named differently in each language but refer to the same law!
        def replace_citations(series, ref_mask_token="<ref>"):
            # TODO think about splitting laws and rulings into two separate labels
            labels = set()
            for law in series.citations['laws']:
                citation = law['text']
                found_string_in_list = string_contains_one_of_list(citation, list(law_abbr_by_lang['de'].keys()))
                if found_string_in_list:
                    series.text = series.text.replace(citation, ref_mask_token)
                    labels.add(law_abbr_by_lang['de'][found_string_in_list])
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

        # Plot the 10 most common citations
        # remove BGG articles because they are obvious
        most_common_interesting = [(k, v) for k, v in most_common_with_frequency if 'BGG' not in k]
        ruling_citations = pd.DataFrame.from_records(most_common_interesting[:plot_n_most_common],
                                                     columns=['citation', 'frequency'])
        ax = ruling_citations.plot.bar(x='citation', y='frequency', rot=90)
        ax.get_figure().savefig(folder / f'most_common_{type}_citations.png', bbox_inches="tight")

        return list(dict(most_common_with_frequency).keys())

    def get_law_abbr_by_lang(self):
        term_definitions = TermDefinitionsExtractor().extract_term_definitions()
        law_abbr_by_lang = {lang: dict() for lang in self.languages}

        for definition in term_definitions:
            for lang in definition['languages']:
                if lang in self.languages:
                    for entry in definition['languages'][lang]:
                        if entry['type'] == 'ab':  # ab stands for abbreviation
                            # append the string of the abbreviation as key and the id as value
                            law_abbr_by_lang[lang][entry['text']] = definition['id']
        return law_abbr_by_lang

    def chamber_prediction(self, engine, lang):
        # TODO process in batches because otherwise too large
        df = self.get_df(engine, 'facts', 'chamber', lang)
        df = df.rename(columns={"facts": "text", "chamber": "label"})  # normalize column names
        labels = list(df.label.unique())
        return df, labels

    def date_prediction(self, engine, lang):
        # TODO process in batches because otherwise too large
        df = self.get_df(engine, 'facts', 'date', lang)
        df = df.rename(columns={"facts": "text", "date": "label"})  # normalize column names
        labels = list(df.label.unique())
        return df, labels

    def get_df(self, engine, feature_col, label_col, lang):
        self.logger.info("Started loading the data from the database")
        origin_canton = "lower_court::json#>>'{canton}' AS origin_canton"
        origin_court = "lower_court::json#>>'{court}' AS origin_court"
        origin_chamber = "lower_court::json#>>'{chamber}' AS origin_chamber"
        columns = f"{feature_col}, {label_col}, extract(year from date) as year, chamber, " \
                  f"{origin_canton}, {origin_court}, {origin_chamber}"
        where = f"{feature_col} IS NOT NULL AND {feature_col} != '' AND {label_col} IS NOT NULL"
        order_by = "year"
        if self.debug:
            chunksize = self.debug_chunksize  # run on smaller dataset for testing
        else:
            chunksize = 2e5
        df = next(self.select(engine, lang, columns=columns, where=where, order_by=order_by, chunksize=int(chunksize)))
        df.year = df.year.astype(int)
        df = self.clean_from_empty_strings(df, feature_col)
        # calculate both the num_tokens for regular words and subwords
        spacy_tokenizer, bert_tokenizer = self.get_tokenizers(lang)
        df['num_tokens_spacy'] = [len(result) for result in spacy_tokenizer.pipe(df[feature_col], batch_size=100)]
        df['num_tokens_bert'] = [len(input_id) for input_id in bert_tokenizer(df[feature_col].tolist()).input_ids]
        df['legal_area'] = df.chamber.apply(get_legal_area)
        df['origin_region'] = df.origin_canton.apply(get_region)
        self.logger.info("Finished loading the data from the database")
        return df

    def get_tokenizers(self, lang):
        if lang == 'de':
            spacy = German()
            bert = "deepset/gbert-base"
        elif lang == 'fr':
            spacy = French()
            bert = "camembert/camembert-base-ccnet"
        elif lang == 'it':
            spacy = Italian()
            bert = "dbmdz/bert-base-italian-cased"
        else:
            raise ValueError(f"Please choose one of the following languages: {self.languages}")
        return spacy.tokenizer, AutoTokenizer.from_pretrained(bert)

    def clean_from_empty_strings(self, df, column):
        # replace empty and whitespace strings with nan so that they can be removed
        df[column] = df[column].replace(r'^\s*$', np.nan, regex=True)
        df = df.dropna(subset=[column])  # drop null values not recognized by sql where clause
        df = df.reset_index(drop=True)  # reindex to get nice indices
        return df

    def save_dataset(self, df: pd.DataFrame, labels: list, folder: Path,
                     split_type="date-stratified", split=(0.7, 0.1, 0.2), kaggle=False):
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
        splits = self.create_splits(df, split, split_type)
        special_splits = self.create_special_splits(split_type, splits)

        self.logger.info(f"Computing metadata reports")
        for name, df in splits.items():
            self.save_report(folder, name, df)

        # save regular dataset
        self.logger.info("Saving the regular dataset")
        self.save_labels(labels, folder / 'labels.json')
        for name, df in splits.items():
            df.to_csv(folder / f'{name}.csv', index_label='id')

        special_splits_dir = self.create_dir(folder, 'special_splits')
        for category, special_split in special_splits.items():
            category_dir = self.create_dir(special_splits_dir, category)
            for name, df in special_split.items():
                df.to_csv(category_dir / f'{name}.csv', index_label='id')

        if kaggle:
            self.logger.info("Saving the data in kaggle format")
            # create solution file
            splits['solution'] = splits['test'].drop('text', axis='columns')  # drop text
            # rename according to kaggle conventions
            splits['solution'] = splits['solution'].rename(columns={"label": "Expected"})
            # create test file
            splits['test'] = splits['test'].drop('label', axis='columns')  # drop label
            # create sampleSubmission file
            # rename according to kaggle conventions
            sample_submission = splits['solution'].rename(columns={"Expected": "Predicted"})
            # set to random value
            sample_submission['Predicted'] = np.random.choice(splits['solution'].Expected, size=len(splits['solution']))
            splits['sample_submission'] = sample_submission

            # save special kaggle files
            kaggle_dir = self.create_dir(folder, 'kaggle')
            for name, df in splits.items():
                df.to_csv(kaggle_dir / f'{name}.csv', index_label='id')

        self.logger.info(f"Saved dataset files to {folder}")

    def create_splits(self, df, split, split_type):
        self.logger.info("Splitting data into train, val and test set")
        if split_type == "random":
            train, val, test = self.split_random(df, split)
        elif split_type == "date-stratified":
            train, val, test = self.split_date_stratified(df, split)
        else:
            raise ValueError("Please supply a valid split_type")
        all = pd.concat([train, val, test])  # we need to update it since some entries have been removed
        return {'train': train, 'val': val, 'test': test, 'all': all}

    def create_special_splits(self, split_type, splits):
        """
        Creates special splits for applications extending beyond the normal splits
        :param split_type:
        :param splits:      the dictionary containing the split dataframes
        :return:
        """
        self.logger.info("Creating special splits")
        test = splits['test']

        special_splits = {
            'input_length': dict(), 'year': dict(), 'legal_area': dict(),
            'origin_region': dict(), 'origin_canton': dict(),
            'origin_court': dict(), 'origin_chamber': dict(),
        }

        # test set split by input length
        boundaries = [0, 512, 1024, 2048, 4096, 8192]
        for i in range(len(boundaries) - 1):
            lower, higher = boundaries[i] + 1, boundaries[i + 1]
            key = f'test-between({lower:04d},{higher:04d})'
            special_splits['input_length'][key] = test[test.num_tokens_bert.between(lower, higher)]

        # test set split by year
        if split_type == "date-stratified":
            for year in range(2017, 2020 + 1):
                special_splits['year'][f'test-{year}'] = test[test.year == year]

        # test set split by legal area
        for legal_area in legal_areas.keys():
            special_splits['legal_area'][f'test-{legal_area}'] = test[test.legal_area.str.contains(legal_area)]

        # test set split by origin region, canton, court and chamber
        region_test = test.dropna(subset=['origin_region'])
        for region in region_test.origin_region.unique().tolist():
            special_splits['origin_region'][f'test-{region}'] = region_test[
                region_test.origin_region.str.contains(region)]

        canton_test = test.dropna(subset=['origin_canton'])
        for canton in canton_test.origin_canton.unique().tolist():
            special_splits['origin_canton'][f'test-{canton}'] = canton_test[
                canton_test.origin_canton.str.contains(canton)]

        court_test = test.dropna(subset=['origin_court'])
        for court in court_test.origin_court.unique().tolist():
            special_splits['origin_court'][f'test-{court}'] = court_test[court_test.origin_court.str.contains(court)]

        chamber_test = test.dropna(subset=['origin_chamber'])
        for chamber in chamber_test.origin_chamber.unique().tolist():
            special_splits['origin_chamber'][f'test-{chamber}'] = chamber_test[
                chamber_test.origin_chamber.str.contains(chamber)]

        return special_splits

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

        self.plot_input_length(df, split_folder)
        self.plot_labels(df, split_folder)

    @staticmethod
    def plot_barplot_attribute(df, split_folder, attribute):
        """
        Plots the distribution of the attribute of the decisions in the given dataframe
        :param df:              the dataframe containing the legal areas
        :param split_folder:    where to save the plots and csv files
        :param attribute:       the attribute to barplot
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

        attribute_df.to_csv(split_folder / f'{attribute}_distribution.csv')

        # TODO refactor this huge class into multiple ones (one for each dataset)

        # TODO make these plots nicer. They are strange now
        attribute_df = attribute_df[~attribute_df[attribute].str.contains('all')]
        ax = sns.barplot(x=attribute, y="number of decisions", data=attribute_df)
        ax.tick_params(labelrotation=30)
        ax.get_figure().savefig(split_folder / f'{attribute}_distribution-histogram.png', bbox_inches="tight")

    @staticmethod
    def plot_labels(df, split_folder):
        """
        Plots the label distribution of the decisions in the given dataframe
        :param df:              the dataframe containing the labels
        :param split_folder:    where to save the plots and csv files
        :return:
        """
        # compute label imbalance
        ax = df.label.astype(str).hist()
        ax.tick_params(labelrotation=30)
        ax.get_figure().savefig(split_folder / 'multi_label_distribution.png', bbox_inches="tight")

        counter_dict = dict(Counter(np.hstack(df.label)))
        counter_dict['all'] = sum(counter_dict.values())
        label_counts = pd.DataFrame.from_dict(counter_dict, orient='index', columns=['num_occurrences'])
        label_counts['percent'] = round(label_counts['num_occurrences'] / counter_dict['all'], 4)
        label_counts.to_csv(split_folder / 'single_label_distribution.csv', index_label='label')

        ax = label_counts[~label_counts.index.str.contains("all")].plot.bar(y='num_occurrences', rot=15)
        ax.get_figure().savefig(split_folder / 'single_label_distribution.png', bbox_inches="tight")

    @staticmethod
    def plot_input_length(df, split_folder):
        """
        Plots the input length of the decisions in the given dataframe
        :param df:              the dataframe containing the decision texts
        :param split_folder:    where to save the plots and csv files
        :return:
        """
        # compute median input length
        input_length_distribution = df[['num_tokens_spacy', 'num_tokens_bert']].describe().round(0).astype(int)
        input_length_distribution.to_csv(split_folder / 'input_length_distribution.csv', index_label='measure')

        # bin outliers together at the cutoff point
        cutoff = 3500
        cut_df = df[['num_tokens_spacy', 'num_tokens_bert']]
        cut_df.loc[cut_df.num_tokens_spacy > cutoff, 'num_tokens_spacy'] = cutoff
        cut_df.loc[cut_df.num_tokens_bert > cutoff, 'num_tokens_bert'] = cutoff

        hist_df = pd.concat([cut_df.num_tokens_spacy, cut_df.num_tokens_bert], keys=['spacy', 'bert']).to_frame()
        hist_df = hist_df.reset_index(level=0)
        hist_df = hist_df.rename(columns={'level_0': 'kind', 0: 'number of tokens'})

        plot = sns.displot(hist_df, x="number of tokens", hue="kind", bins=50, kde=True, fill=True)
        plot.savefig(split_folder / 'input_length_distribution-histogram.png', bbox_inches="tight")

        plot = sns.displot(hist_df, x="number of tokens", hue="kind", kind="ecdf")
        plot.savefig(split_folder / 'input_length_distribution-cumulative.png', bbox_inches="tight")

        plot = sns.displot(cut_df, x="num_tokens_spacy", y="num_tokens_bert", cbar=True)
        plot.savefig(split_folder / 'input_length_distribution-bivariate.png', bbox_inches="tight")

    @staticmethod
    def save_labels(labels, file_name):
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

    @staticmethod
    def split_date_stratified(df, split):
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
