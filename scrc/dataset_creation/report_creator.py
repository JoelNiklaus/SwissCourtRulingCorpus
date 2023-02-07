import abc
from typing import Dict
import math
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from pathlib import Path
import numpy as np
import os

from scrc.utils.log_utils import get_logger

from scrc.utils.sql_select_utils import get_legal_area_bger


class ReportCreator:
    """
    Creates all kind of plots for a given dataframe
    """

    def __init__(self, base_folder, debug):
        self.folder: Path = base_folder
        self.logger = get_logger(__name__)
        self.debug = debug

    def plot_attribute(self, df, attribute, name=""):
        """
        Plots the distribution of the attribute of the decisions in the given dataframe
        :param df:              the dataframe containing the legal areas
        :param self.folder:          specifies where to save
        :param attribute:       the attribute to barplot
        :param name:            name of the plot
        :return:
        """
        if not df.empty:
            attribute_df = df[attribute].value_counts().to_frame()
            total = len(df.index)
            # we deleted the ones where we did not find any attribute: also mention them in this table
            uncategorized = total - attribute_df[attribute].sum()
            attribute_df = attribute_df.reset_index(level=0)
            attribute_df = attribute_df.rename(columns={'index': attribute, attribute: 'number of decisions'})
            attribute_df['number of decisions'] = attribute_df['number of decisions'].astype(int)
            attribute_df.sort_values(by=[attribute], inplace=True)
            if uncategorized != 0:
                attribute_df.loc[len(attribute_df.index)] = ['uncategorized', uncategorized]
            attribute_df.loc[len(attribute_df.index)] = ['all', total]
            attribute_df['percent'] = round(attribute_df['number of decisions'] / total, 4)
            attribute_df.to_csv(self.folder / f'{name}_{attribute}_distribution.csv')

            # get rid of unnecessary rows / cols
            if attribute == 'language':
                attribute_df = attribute_df[~attribute_df[attribute].astype(str).str.contains('en')]
                attribute_df = attribute_df[~attribute_df[attribute].astype(str).str.contains('--')]
            attribute_df = attribute_df[~attribute_df[attribute].astype(str).str.contains('all')]

            # TODO order attributes?

            fig = px.bar(attribute_df, x=attribute, y="number of decisions",
                         title=f'{name} {attribute}')
            fig.write_image(self.folder / f'{name}_{attribute}_distribution-histogram.png')
            plt.close()

    def plot_label_ordered(self, df, label_name, order=dict()):
        """
        Plots the label distribution of the decisions in the given dataframe
        :param df:              the dataframe containing the labels
        :param label_name:       name of attribute
        :param order:           specifies a certain order
        :return:
        """

        if not df.empty:
            counter_dict = dict(Counter(np.hstack(df.label)))
            counter_dict['all'] = sum(counter_dict.values())
            label_counts = pd.DataFrame.from_dict(counter_dict, orient='index', columns=['num_occurrences'])
            label_counts.loc[:, 'percent'] = round(label_counts['num_occurrences'] / counter_dict['all'], 4)
            label_counts.to_csv(self.folder / f"{label_name}_distribution.csv", index_label='label')

            if order:
                ax = label_counts[~label_counts.index.str.contains("all")].plot.bar(y='num_occurrences', rot=15)
            else:
                ax = label_counts[~label_counts.index.str.contains("all")].plot.bar(y='num_occurrences', rot=15)
            ax.get_figure().savefig(self.folder / f"{label_name}_distribution.png", bbox_inches="tight")
            plt.close()

    def plot_attribute_color(self, df, attribute, color_attribute, name):
        """
               Plots the distribution of the attribute of the decisions in the given dataframe
               :param df:              the dataframe containing the legal areas
               :param attribute:       the attribute to barplot
               :param color_attribute:
               :param name:
               :return:
               """
        fig = px.histogram(df, x=attribute, color=color_attribute)
        fig.write_image(self.folder / f'{name}_{attribute}_{color_attribute}_colored_histogram.png')
        plt.close()

    def plot_input_length(self, df, feature_col):
        """
        Plots the input length of the decisions in the given dataframe
        :param df:              the dataframe containing the decision texts
        :param feature_col:     specifies feature_col
        :return:
        """
        # compute median input length
        input_length_distribution = df.loc[:, ['num_tokens_spacy', 'num_tokens_bert']].describe().round(0)
        if len(df.index) == 1:
            input_length_distribution = input_length_distribution.fillna(df.mean())
        input_length_distribution = input_length_distribution.astype(int)
        input_length_distribution.to_csv(self.folder / f'{feature_col}_input_length_distribution.csv',
                                         index_label='measure')

        # bin outliers together at the cutoff point
        cutoff = 4000
        cut_df = df.loc[:, ['num_tokens_spacy', 'num_tokens_bert']]
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
        plot.savefig(self.folder / f'{feature_col}_input_length_distribution-histogram.png', bbox_inches="tight")
        plt.close()

        plot = sns.displot(hist_df, x="Number of tokens", hue="tokenizer", kind="ecdf", legend=False)
        plt.ylabel('Number of court cases')
        plt.legend(["BERT", "SPaCy"], loc='lower right', title='Tokenizer')
        plot.savefig(self.folder / f'{feature_col}_input_length_distribution-cumulative.png', bbox_inches="tight")
        plt.close()

        plot = sns.displot(cut_df, x="num_tokens_spacy", y="num_tokens_bert")
        plot.savefig(self.folder / f'{feature_col}_input_length_distribution-bivariate.png', bbox_inches="tight")
        plt.close()

    def plot_two_attributes(self, df, x_attribute, y_attribute, name, how='scatter'):
        """
        Create a plot comparing two attributes with each other
        :param df:          dataframe with all data
        :param x_attribute: attribute on x-axis
        :param y_attribute: attribute on y-axis
        :pararm name:
        :param how:         specifies if scatter or histogram is created
        """
        # TODO should both attributes be numeric?
        if how == 'scatter':
            plot = df.plot(x=x_attribute, y=y_attribute, kind='scatter', title=f'{x_attribute} {y_attribute}')
            fig = plot.get_figure()
            fig.savefig(self.folder / f'{name}_{x_attribute}_on_{y_attribute}_scatter.png')
            plt.close()
        else:
            fig = px.bar(df, x=x_attribute, y=y_attribute,
                         title=f'{name} {x_attribute} {y_attribute}')
            fig.write_image(self.folder / f'{name}_{x_attribute}_on_{y_attribute}_plot.png')
            plt.close()

    def bin_plot_attribute(self, df, attribute, color_attribute, bin_start, bin_end, bin_steps):
        """
        Plots the distribution of the attribute of the decisions in the given dataframe
        :param df:              the dataframe containing the legal areas
        :param attribute:       the attribute to barplot
        :param color_attribute: attribute which is used to color the graph
        :param bin_start:
        :param bin_end:
        :param bin_steps:
        :return:
        """
        df = df[df[attribute] < bin_end]
        counts, bins = np.histogram(df[attribute], bins=range(bin_start, bin_end, bin_steps))
        bins = 0.5 * (bins[:-1] + bins[1:])
        fig = px.bar(x=bins, y=counts, labels={'x': attribute, 'y': 'number of cases'})
        fig.write_image(self.folder / f'{attribute}_bins.png')

        bin_amount = int(round((bin_end - bin_start) / bin_steps, 0))
        fig = px.histogram(df, x=attribute, nbins=bin_amount, color=color_attribute)
        fig.write_image(self.folder / f'{attribute}_bins_{color_attribute}.png')

    def report_general(self, metadata, feature_cols, labels, df):
        """
        Saves statistics and reports about bge_label and citation_label. Specific for criticality_dataset_creator
        :param metadata:
        :param df:              the df containing the dataset
        """
        for attribute in metadata:
            if labels:
                for label in labels:
                    match = df[label] == 'non-critical'
                    try:
                        self.plot_attribute(df[~match], attribute, name=str(label))
                    except:
                        self.logger.info(f'Could not plot {attribute} for {label}. (Ignore if this is {attribute} dataset)')
                        continue
            else:
                self.plot_attribute(df, attribute, name='all')

        if 'origin_facts' in df.columns:
            feature_cols.extend(['origin_facts', 'origin_considerations'])
            for feature_col in feature_cols:
                df[feature_col] = df[feature_col].replace('', np.nan)
                # drop all rows with NaN in these columns
                df = df.dropna(subset=[feature_col])

        for feature_col in feature_cols:
            tokens_dict: Dict[str, str] = {f'{feature_col}_num_tokens_bert': 'num_tokens_bert',
                    f'{feature_col}_num_tokens_spacy': 'num_tokens_spacy'}
            try:
                if len(df) > 0:
                    self.plot_input_length(df.rename(columns=tokens_dict), feature_col)
            except np.linalg.LinAlgError as err:
                if 'singular matrix' in str(err):
                    print("Singular matrix error in plot_input_length")
                else:
                    raise err

    def report_citations_count(self, df, name):
        """
        Saves reports about each citation_label class.
        This needs to be done plot_custom because count to cover all found citations independent whether bge is in db or not
        :param df:      dataframe containing all the data
        """
        # report distribution of citation, here because it's deleted from df later.

        self.plot_attribute(df, 'year', name=str(name))
        self.plot_attribute(df, 'bge_chamber', name=str(name))
        self.plot_attribute(df, 'bger_chamber', name=str(name))

        if name == 'all':
            self.report_citations_details(df)

    def report_citations_details(self, df):
        df.to_csv(self.folder / f'citation_distribution.csv')

        self.plot_two_attributes(df, 'year', 'counter', 'all')
        self.bin_plot_attribute(df, 'counter', 'bge_chamber', 0, 300, 10)
        self.bin_plot_attribute(df, 'counter', 'bger_chamber', 0, 300, 10)
        self.bin_plot_attribute(df, 'counter', 'law_area', 0, 300, 10)

        my_dictionary_1 = dict.fromkeys(list(range(0, 301, 10)))
        my_dictionary_2 = dict.fromkeys(list(range(0, 51, 1)))
        my_dictionary_1 = dict(zip(my_dictionary_1, [0 for _ in my_dictionary_1]))
        my_dictionary_2 = dict(zip(my_dictionary_2, [0 for _ in my_dictionary_2]))

        def roundup_10(x):
            return int(math.ceil(float(x) / 10.0)) * 10
        for index, row in df.iterrows():
            counter = row['counter']
            counter_10 = roundup_10(counter)
            counter = int(math.ceil(float(counter)))
            if int(counter) <= 50:
                my_dictionary_2[int(counter)] = my_dictionary_2[int(counter)] + counter
            if counter_10 > 300:
                counter_10 = 300
            my_dictionary_1[counter_10] = my_dictionary_1[counter_10] + counter_10

        data = {'counter': list(my_dictionary_1.keys()), 'number of decisions': my_dictionary_1.values()}
        citations_amount_df = pd.DataFrame(data)
        self.plot_two_attributes(citations_amount_df, 'counter', 'number of decisions', 'citations', how='histogram')

        data = {'counter': list(my_dictionary_2.keys()), 'number of decisions': my_dictionary_2.values()}
        citations_amount_df = pd.DataFrame(data)
        self.plot_two_attributes(citations_amount_df, 'counter', 'number of decisions', 'citations', how='histogram')

    def report_references(self, df):
        plot_attributes = ['bge_chamber', 'law_area', 'bger_chamber', 'year']
        for attribute in plot_attributes:
            self.plot_attribute(df, attribute, name='references')
        self.plot_attribute_color(df, 'year', 'bge_chamber', 'references')
        self.plot_attribute_color(df, 'year', 'law_area', 'references')

    def report_references_not_found(self, not_found_list, label):
        """
        Save list of all bger_references which were extracted but could not be found as bge in db
        :param not_found_list:  list of references
        :param label:           specifying for with label class the given list was created for
        """
        updated_list = list(set(not_found_list))
        file_path = self.folder / "not_found_references.txt"
        with open(file_path, 'w+') as f:
            f.write(f"List of not found references for {label}:\n")
            for item in updated_list:
                f.write(f"{item}\n")
        # report
        df = pd.DataFrame({'chamber': [], 'year': [], 'legal_area': []})
        for item in updated_list:
            chamber = item.split('_', 2)[0]
            legal_area = get_legal_area_bger(chamber)
            # TODO get nice representation of chamber
            chamber = chamber[0]
            year = int(item.split('/', 2)[1])
            s_row = pd.Series([chamber, year, legal_area], index=df.columns)
            df = df.append(s_row, ignore_index=True)
        df = df[df['year'] > 2000]
        df = df[df['year'] < 2030]
        self.plot_attribute_color(df, 'year', 'legal_area', 'references_not_found')

    def test_correctness_of_labeling(self, not_found_list, references_df):
        """
        in order to assure correctness, test which cases could not be found and why
        :param not_found_list:      list of references were no bger case could be found for.
        """
        # get list where bge file number is multiple times in references df
        file_number_list = references_df.bge_file_number_long.tolist()
        file_number_counter = Counter(file_number_list)
        double_bge_file_numbers = [k for k, v in file_number_counter.items() if v > 1]
        # get list where bger references is multiple time in references df
        reference_list = references_df.bger_reference.tolist()
        reference_counter = Counter(reference_list)
        double_bger_references = [k for k, v in reference_counter.items() if v > 1]
        # find cases where bge_reference and file number did occur mutliple times
        file_number_match = references_df.bge_file_number_long.astype(str).isin(list(double_bge_file_numbers))
        doubled_df = references_df[file_number_match]
        file_number_match = doubled_df.bger_reference.astype(str).isin(list(double_bger_references))
        doubled_df = doubled_df[file_number_match]
        # make sure an entry in not_found_list is not wrongly found as entry in references
        for item in not_found_list:
            assert item not in file_number_list
        # check how many cases of the not found list is from before 2000
        file_number_match = references_df.bger_reference.astype(str).isin(not_found_list)
        not_found_df = references_df[file_number_match]
        new_cases = not_found_df.loc[not_found_df['year'] > 2000]
        print(f"Not_found_list: there are {len(new_cases)} cases")
