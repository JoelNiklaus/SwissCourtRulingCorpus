import configparser
import inspect
from collections import Counter

from bs4 import BeautifulSoup

from root import ROOT_DIR
from scrc.data_classes.law_citation import LawCitation
from scrc.data_classes.ruling_citation import RulingCitation
from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.utils.log_utils import get_logger
import numpy as np
import pandas as pd

from scrc.utils.main_utils import get_config, string_contains_one_of_list
from scrc.utils.term_definitions_converter import TermDefinitionsConverter


class CitationDatasetCreator(DatasetCreator):
    """
    Creates a dataset for information retrieval with the citations serving as ground truth for relevance scores
    """

    # usefulness of law articles still unclear.
    # Approach of Lawyers:
    # google
    # get law articles
    # get relevant case law
    #
    # what they want is:
    # - from facts of a case get other cases
    # - from law article get cases that cite this article ==> starting with newest case (lawyer can follow back)

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.debug = False
        self.split_type = "date-stratified"
        self.dataset_name = "doc2doc_ir"
        self.feature_cols = ['facts', 'considerations']

        self.num_ruling_citations = 1000  # the 1000 most common ruling citations will be included

        self.load_rulings()
        self.load_law_articles()

    def load_rulings(self):
        self.logger.info(f"Loading reference rulings")
        self.available_bges = set()  # use set instead of list for faster lookup
        rulings = pd.DataFrame()
        for lang in self.languages:
            # also for debugging we want the full list
            cols = "language, canton, date, file_number, html_url, text"
            df = next(self.select(self.get_engine(self.db_scrc), lang,
                                  columns=cols, where="spider = 'CH_BGE'", chunksize=self.real_chunksize))
            df.date = pd.to_datetime(df.date)
            assert len(df.index) == len(df.file_number.unique())  # there should not be any duplicates
            self.available_bges.update(df.file_number.tolist())
            rulings = pd.concat([rulings, df], ignore_index=True)  # one rulings df for all languages

        rulings_path = self.create_dir(self.datasets_subdir, self.dataset_name) / "rulings.jsonl"
        if not rulings_path.exists():
            rulings.to_json(rulings_path, orient="records", lines=True, date_format="iso")

    def load_law_articles(self):
        # get Art. from lexfind.jsonl
        self.logger.info(f"Loading reference law articles")
        laws = pd.read_json(self.corpora_subdir / "lexfind.jsonl", lines=True)
        laws = laws[laws.canton.str.contains("ch")]  # only federal laws so far
        laws = laws[laws.abbreviation.str.len() > 1]  # only keep the ones with an abbreviation
        self.available_laws = set(laws.abbreviation.unique().tolist())

        articles_path = self.create_dir(self.datasets_subdir, self.dataset_name) / "articles.jsonl"
        if not articles_path.exists():
            self.logger.info("Extracting individual articles from laws")
            articles = pd.concat([self.extract_article_info(row) for _, row in laws.iterrows()], ignore_index=True)
            articles.to_json(articles_path, orient="records", lines=True)

    @staticmethod
    def extract_article_info(law):
        """Extracts the information for the individual articles from a given law"""
        arts = {"canton": [], "language": [], "sr_number": [], "abbreviation": [], "short": [], "title": [],
                "link": [], "text": [], }
        bs = BeautifulSoup(law.html_content, "html.parser")
        for art in bs.find_all("article"):
            arts["canton"].append(law.canton)
            arts["language"].append(law.language)
            arts["sr_number"].append(law.sr_number)
            arts["abbreviation"].append(law.abbreviation)
            arts["short"].append(law.short)
            arts["title"].append(law.title)

            link = art.find(fragment=f"#{art['id']}")
            if link:
                arts["link"].append(link['href'])
            else:
                arts["link"].append(None)
            text = art.find("div", {"class": "collapseable"})
            if text:
                arts["text"].append(text.get_text())
            else:
                arts["text"].append(None)
        return pd.DataFrame(arts)

    def get_dataset(self, feature_col, lang, save_reports):
        df = self.get_df(self.get_engine(self.db_scrc), feature_col, 'citations', lang, save_reports)

        # df['types'] = df.citations.apply(lambda x: np.unique([cit['type'] for cit in x['rulings']]))
        # df = df[df.types.map(len) >= 1]  # we only want the decisions which actually cite something
        # df = df.query('"bge" in types')  # ensure that we only have BGE citations

        df['rulings'] = df.citations.apply(self.get_ruling_citations)
        df['laws'] = df.citations.apply(self.get_law_citations)
        df = df.dropna(subset=["rulings", "laws"], how="all")  # we cannot use the ones which have no citations

        relevance_lambda = lambda cits: [(cit_tuple[0], self.compute_relevance_score(cit_tuple))
                                         for cit_tuple in cits] if cits else None
        df.rulings = df.rulings.apply(relevance_lambda)
        df.laws = df.laws.apply(relevance_lambda)

        df = df.drop(['citations'], axis=1)  # we don't need this anymore

        # TODO extract ruling citations from other decisions and test the extraction regexes on the CH_BGer

        # this_function_name = inspect.currentframe().f_code.co_name
        folder = self.create_dir(self.datasets_subdir, self.dataset_name)

        # TODO move this to the plot_custom function so that we can save it for all languages

        # calculate most common BGE citations
        most_common_rulings = self.get_most_common_citations(df, folder, 'rulings')

        # calculate most common laws
        most_common_laws = self.get_most_common_citations(df, folder, 'laws')

        #  IMPORTANT: we need to take care of the fact that the laws are named differently in each language but refer to the same law!
        law_abbr_by_lang = self.get_law_abbr_by_lang()

        def mask_citations(series, law_mask_token="<ref-law>", ruling_mask_token="<ref-ruling>",
                           only_replace_most_common_citations=False):
            """
            Mask citations so that they don't help in finding the right document,
            but it should only use the context
            """
            for law in series.citations['laws']:
                if only_replace_most_common_citations \
                        and not string_contains_one_of_list(law['text'], list(law_abbr_by_lang[lang].keys())):
                    continue
                series[feature_col] = series[feature_col].replace(law['text'], law_mask_token)
            for ruling in series.citations['rulings']:
                if only_replace_most_common_citations \
                        and not string_contains_one_of_list(ruling['text'], most_common_rulings):
                    continue
                series[feature_col] = series[feature_col].replace(ruling['text'], ruling_mask_token)
            return series

        df = df.apply(mask_citations, axis='columns')
        df = df.rename(columns={feature_col: "text"})  # normalize column names
        df['label'] = df.rulings  # the label is just the rulings for now
        return df, self.available_bges

    def get_ruling_citations(self, citations):
        # get BGEs by file_number
        cits = []
        for citation in citations['rulings']:
            cit = citation['text']
            cit = ' '.join(cit.split())  # remove multiple whitespaces inside
            try:
                ruling_cit = RulingCitation(cit)
            except ValueError as ve:
                self.logger.warning(ve)
                continue
            # only actually include citations that we can find in our corpus
            if str(ruling_cit) in self.available_bges:
                cits.append(ruling_cit)
        if cits:  # only return something if we actually have citations
            return list(Counter(cits).items())

    def get_law_citations(self, citations):
        cits = []
        for citation in citations['laws']:
            cit = citation['text']
            cit = ' '.join(cit.split())  # remove multiple whitespaces inside
            try:
                law_cit = LawCitation(cit)
            except ValueError as ve:
                self.logger.warning(ve)
                continue
            # only actually include citations that we can find in our corpus
            if law_cit.abbreviation in self.available_laws:
                cits.append(law_cit)
        if cits:  # only return something if we actually have citations
            return list(Counter(cits).items())

    @staticmethod
    def compute_relevance_score(citation_tuple):
        """
        Computes a relevance score for the citation between 1 and 1
            frequency: 70%
            year: 30%
        :return:
        # Proxy für relevanz score:
        #   - häufigkeit des Zitats in Erwägungen
        #   - neuere Urteile sind relevanter (ältere Urteile sind evtl. überholt oder nicht mehr relevant)
        #   Thomas schreibt mir seine Ideen bald
        #   - BGE wichtiger als andere ==> nur BGEs nehmen
        """
        weights = {"frequency": 0.7, "year": 0.3}
        assert sum(weights.values()) == 1

        values = {"frequency": citation_tuple[1] / 10}
        citation = citation_tuple[0]
        if isinstance(citation, RulingCitation):
            # 150 corresponds to the year 2024 which should give use enough leeway
            values["year"] = citation.year / 150
        else:
            values["year"] = 0.5  # set it to middle importance for all law citations

        # compute weighted sum
        score = 0
        for key in weights.keys():
            score += weights[key] * values[key]

        return round(min(score, 1), 2)  # cap it at 1 and round to 2 decimals

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
    config = get_config()

    citation_dataset_creator = CitationDatasetCreator(config)
    citation_dataset_creator.create_dataset(sub_datasets=False, kaggle=False, huggingface=True, save_reports=False)
