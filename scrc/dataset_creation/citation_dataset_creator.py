import configparser
import inspect
from collections import Counter

from root import ROOT_DIR
from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.utils.log_utils import get_logger
import numpy as np
import pandas as pd

from scrc.utils.main_utils import string_contains_one_of_list
from scrc.utils.term_definitions_converter import TermDefinitionsConverter


class CitationDatasetCreator(DatasetCreator):
    """
    Creates a dataset with the text as input and the citations as labels
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.debug = False
        self.split_type = "date-stratified"
        self.dataset_name = "citation_prediction"

        self.num_ruling_citations = 1000  # the 1000 most common ruling citations will be included


    def get_dataset(self, feature_col, lang, save_reports):
        df = self.get_df(self.get_engine(self.db_scrc), feature_col, 'citations', lang, save_reports)

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
                found_string_in_list = string_contains_one_of_list(citation, list(law_abbr_by_lang[lang].keys()))
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
        term_definitions = TermDefinitionsConverter().extract_term_definitions()
        law_abbr_by_lang = {lang: dict() for lang in self.languages}

        for definition in term_definitions:
            for lang in definition['languages']:
                if lang in self.languages:
                    for entry in definition['languages'][lang]:
                        if entry['type'] == 'ab':  # ab stands for abbreviation
                            # append the string of the abbreviation as key and the id as value
                            law_abbr_by_lang[lang][entry['text']] = definition['id']
        return law_abbr_by_lang


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    citation_dataset_creator = CitationDatasetCreator(config)
    citation_dataset_creator.create_dataset()
