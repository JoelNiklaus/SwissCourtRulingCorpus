import itertools
from collections import Counter
import ast

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

from scrc.data_classes.law_citation import LawCitation
from scrc.data_classes.ruling_citation import RulingCitation
from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.utils.log_utils import get_logger
import numpy as np
import pandas as pd
from pandarallel import pandarallel

from scrc.utils.main_utils import get_config, string_contains_one_of_list

"""
BGE volumes
Bände und Sachgebiete bis 1994 (Jahrgänge 111 bis 120):
Ia Verfassungsrecht
Ib Verwaltungsrecht und Internationales öffentliches Recht
II Zivilrecht
III Betreibungs- und Konkursrecht
IV Strafrecht und Strafvollzug
V Sozialversicherungsrecht
Bände und Sachgebiete seit 1995 (Jahrgang 121):
I Verfassungsrecht
II Verwaltungsrecht und Internationales öffentliches Recht
III Zivilrecht, Schuldbetreibungs- und Konkursrecht
IV Strafrecht und Strafvollzug
V Sozialversicherungsrecht
"""

"""
Multiple possible tasks:
Two collections: rulings and law articles
- hard task: IR task with specific law article
- easy task: IR task with only laws
- multiple choice qa: instead of whole collection of rulings/laws: give 5 options, one relevant and 4 irrelevant
Story: we want to solve hard task, but we might be very bad at it yet: propose easier tasks to make progress
Experiments:
are we better in different legal areas / regions etc.
TODO:
how long are the documents
how many citations do the documents have (for both laws and rulings) ==> histogram
distribution of the time difference between published document and cited document
try to find out how multilingual this corpus is: how many times does a German decision cite an Italian ruling?
find out which laws are cited most often: ask lawyers to classify laws into legal areas
train legal area classifier: ==> classify BGEs into legal areas automatically (Dominik)
Frage an Thomas: werden verschiedene BGE Bände zitiert in einem Urteil?
Antwort: Ja
"""


# TODO Compute statistics for sections: how many found, token length distributions
# TODO Use Codalab for next benchmark paper: https://codalab.org/
# TODO add uuid to the dataset export for easy identification
# TODO consider using haystack/FARM for IR experiments

class Doc2DocIRDatasetCreator(DatasetCreator):
    """
    Creates a dataset for information retrieval with the citations serving as ground truth for relevance scores
    """

    """
    Dataset description:
    documents:
        laws.jsonl:
            contains all the law articles that can be retrieved (one article per line)
        rulings.jsonl:
            contains all the reference rulings that can be retrieved (one ruling per line)
    queries:
        queries.jsonl:
            contains the court decisions that are fed into the model as queries (one query per line)
            the decisions contain metadata such as language, date, chamber, origin region
            the labels (relevant documents) are listed in the fields "rulings_relevanes" and "laws_relevances" with the number (0 for lowest relevance - 1 for highest relevance) representing the relevance of the given document
            laws can be matched by the sr-number and the article number and rulings can be matched by the year, volume and page number
        train.jsonl:
            contains only the train split of queries.jsonl (years 2000 - 2014)
        val.jsonl:
            contains only the validation split of queries.jsonl (years 2015 - 2016)
        test.jsonl:
            contains only the test split of queries.jsonl (years 2017 - 2021)

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

        self.debug = True
        self.split_type = "date-stratified"
        self.dataset_name = "doc2doc_ir"
        self.feature_cols = ['facts', 'considerations']

        self.dataset_folder = self.create_dir(self.datasets_subdir, self.dataset_name)
        if self.debug:
            # make sure that we don't overwrite progress in the real directory
            self.dataset_folder = self.create_dir(self.tmp_subdir, self.dataset_name)

        self.num_ruling_citations = 1000  # the 1000 most common ruling citations will be included

        self.load_rulings()
        self.load_law_articles()

        pandarallel.initialize(progress_bar=True)
        tqdm.pandas()

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

        self.logger.info(f"Found {len(self.available_bges)} rulings")
        rulings_path = self.dataset_folder / "rulings.jsonl"
        if not rulings_path.exists():
            rulings.to_json(rulings_path, orient="records", lines=True, date_format="iso")

    def load_law_articles(self):
        # get Art. from lexfind.jsonl
        self.logger.info(f"Loading reference law articles")
        laws = pd.read_json(self.corpora_subdir / "lexfind.jsonl", lines=True)
        # TODO change this, as soon as we include other courts
        laws = laws[laws.canton.str.contains("ch")]  # only federal laws so far
        laws = laws[laws.abbreviation.str.len() > 1]  # only keep the ones with an abbreviation
        laws.abbreviation = laws.abbreviation.str.strip()

        self.law_abbrs = laws[["language", "abbreviation", "sr_number"]]
        self.logger.info(self.law_abbrs.abbreviation.tolist())

        # TODO think about if we need to combine the laws in the different languages to one single law with representations in different languages
        self.available_laws = set(laws.abbreviation.unique().tolist())
        self.logger.info(f"Found {len(self.available_laws)} laws")
        laws_path = self.dataset_folder / "laws.jsonl"
        # This can take quite a long time
        if not laws_path.exists():
            self.logger.info("Extracting individual articles from laws")
            articles = pd.concat([self.extract_article_info(row) for _, row in laws.iterrows()], ignore_index=True)
            articles.to_json(laws_path, orient="records", lines=True)

    @staticmethod
    def extract_article_info(law):
        """Extracts the information for the individual articles from a given law"""
        arts = {"canton": [], "language": [], "sr_number": [], "abbreviation": [], "article": [], "short": [],
                "title": [],
                "link": [], "text": [], }
        bs = BeautifulSoup(law.html_content, "html.parser")
        for art in bs.find_all("article"):
            arts["canton"].append(law.canton)
            arts["language"].append(law.language)
            arts["sr_number"].append(law.sr_number)
            arts["abbreviation"].append(law.abbreviation)
            arts["short"].append(law.short)
            arts["title"].append(law.title)

            article_number = art['id'].split("_")[1]
            arts["article"].append(article_number)

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

    def get_dataset(self, feature_col, save_reports):
        df = self.get_df(self.get_engine(self.db_scrc), feature_col)

        # df['types'] = df.citations.apply(lambda x: np.unique([cit['type'] for cit in x['rulings']]))
        # df = df[df.types.map(len) >= 1]  # we only want the decisions which actually cite something
        # df = df.query('"bge" in types')  # ensure that we only have BGE citations

        df = self.process_citation_type("rulings", df)
        df = self.process_citation_type("laws", df)

        df = df.apply(self.mask_citations, feature_col=feature_col, axis='columns')

        # we don't need this anymore
        df = df.drop(['citations', 'counter', 'rulings', 'laws'], axis=1)

        # TODO don't save here, give this back to the dataset_creator
        query_path = self.dataset_folder / "queries.jsonl"
        self.logger.info(f"Saving queries to {query_path}")
        df.to_json(query_path, orient="records", lines=True)

        # TODO check that the law keys match from the articles.jsonl file

        exit()

        # TODO extract ruling citations from other decisions and test the extraction regexes on the CH_BGer

        df = df.rename(columns={feature_col: "text"})  # normalize column names
        df['label'] = df.rulings  # the label is just the rulings for now
        return df, self.available_bges

    @staticmethod
    def mask_citations(series, feature_col, law_mask_token="<ref-law>", ruling_mask_token="<ref-ruling>"):
        """
        Mask citations so that they don't help in finding the right document,
        but it should only use the context
        """
        for law in series.citations['laws']:
            series[feature_col] = series[feature_col].replace(law['text'], law_mask_token)
        for ruling in series.citations['rulings']:
            series[feature_col] = series[feature_col].replace(ruling['text'], ruling_mask_token)
        return series

    def process_citation_type(self, cit_type, df):
        self.logger.info(f"Processing the {cit_type} citations.")
        df.loc[df['lang'] == 'de', 'ruling_citation'] = df.citations.apply(self.get_citations, type=cit_type, lang='de')
        df.loc[df['lang'] == 'fr', 'ruling_citation'] = df.citations.apply(self.get_citations, type=cit_type, lang='fr')
        df.loc[df['lang'] == 'it', 'ruling_citation'] = df.citations.apply(self.get_citations, type=cit_type, lang='it')
        # we cannot use the ones which have no citations
        # because we drop everything we lose some data, but it is easier, because we know that all the entries have both laws citations and rulings citations
        # reset the index so that we don't get out of bounds errors
        df = df.dropna(subset=[cit_type]).reset_index(drop=True)

        self.logger.info(f"Building the term-frequency matrix.")
        corpus = df[cit_type].tolist()
        vocabulary = sorted(list(set(itertools.chain(*corpus))))
        df[f"counter"] = df[cit_type].apply(lambda x: dict(Counter(x)))
        tf = self.build_tf_matrix(corpus, vocabulary, df)

        self.logger.info(f"Calculating the inverse document frequency scores.")
        tf_idf = TfidfTransformer().fit_transform(tf)

        self.logger.info("Computing relevance scores for documents in collection")
        # x.name retrieves the row index
        relevance_lambda = lambda x: {cit: self.compute_relevance_score(cit, tf_idf[x.name, vocabulary.index(cit)])
                                      for cit, _ in x[f"counter"].items()}
        df[f"{cit_type}_relevances"] = df.parallel_apply(relevance_lambda, axis=1)

        # TODO move this to the plot_custom function so that we can save it for all languages
        self.logger.info("Saving graphs about the dataset.")
        type_corpus_frequencies = dict(Counter(itertools.chain(*df[cit_type].tolist())))
        self.save_citations(type_corpus_frequencies, cit_type)
        return df

    @staticmethod
    def build_tf_matrix(corpus, vocabulary, type_df):
        # build term frequency matrix
        tf = np.zeros((len(corpus), len(vocabulary)))  # term frequencies: number of documents x number of citations

        def fill_tf_matrix(x):
            for cit, count in x[f"counter"].items():
                tf[x.name, vocabulary.index(cit)] = count

        type_df.progress_apply(fill_tf_matrix, axis=1)

        # TODO can be deleted
        # for doc_index, counter in tqdm(type_counter.iteritems(), total=len(corpus)):
        #    for cit, count in counter.items():
        #        cit_index = vocabulary.index(cit)
        #        tf[doc_index][cit_index] = count
        return tf

    def save_citations(self, citation_frequencies: dict, name: str, top_n=25):
        figure_path = self.dataset_folder / f"{name}_citations.png"
        csv_path = self.dataset_folder / f"{name}_citations.csv"

        citations = pd.DataFrame.from_dict(citation_frequencies, orient="index", columns=["frequency"])
        citations = citations.sort_values(by=['frequency'], ascending=False)
        citations.to_csv(csv_path)

        ax = citations[:top_n].plot.bar(use_index=True, y='frequency', rot=90)
        ax.get_figure().savefig(figure_path, bbox_inches="tight")

    def get_citations(self, citations_as_string, type, lang):
        # TODO this code will not work as intended:
        #  - iteration over all citation types, additional check neccessarry.
        #  - all file_numbers in available_bges start with BGE never ATF or DTF
        #  - conseder using one method get_citations for criticality and doc2doc
        cits = []
        try:
            citations = ast.literal_eval(citations_as_string)  # parse dict string to dict again
            for citation in citations:
                cit = citation['text']
                cit = ' '.join(cit.split())  # remove multiple whitespaces inside
                try:
                    if type == "rulings":
                        type_cit = RulingCitation(cit, lang)
                    elif type == "laws":
                        type_cit = LawCitation(cit, lang, self.law_abbrs)
                    else:
                        raise ValueError("type must be either 'rulings' or 'laws'")
                except ValueError as ve:
                    self.logger.debug(ve)
                    continue
        except ValueError as ve:
            self.logger.debug(ve)
        # only actually include ruling citations that we can find in our corpus
        if type == "laws" or (type == "rulings" and str(type_cit) in self.available_bges):
            cits.append(type_cit)

        if cits:  # only return something if we actually have citations
            return cits

    @staticmethod
    def compute_relevance_score(cit, tf_idf_score):
        """
        Computes a relevance score for the citation between 0 and 1
        Algorithm:
        1. tf_idf score for citation: frequency of citation in the considerations (==> main question) *
         inverse frequency of citation in corpus (over all splits for code simplicity) (==> most specific citation)
        2. maybe consider criticality score from Ronja
        :return:
        # Proxy für relevanz score:
        #   - tf für zitierungen: häufigkeit des Zitats in Erwägungen
        #   - idf für zitierungen: wenn Zitat im Korpus häufig vorkommt hat es tiefe Relevanz ==> neuere BGEs kommen weniger häufig vor und sind somit wichtiger
        #   - neuere Urteile sind relevanter (ältere Urteile sind evtl. überholt oder nicht mehr relevant), aber es hängt von der Frage ab (pro rechtlicher Frage wäre das neuere BGE relevanter)
        #   - BGE wichtiger als andere ==> nur BGEs nehmen
        Future work: investigate IR per legal question
        """
        # TODO add Ronja's criticality score
        return tf_idf_score


if __name__ == '__main__':
    config = get_config()

    citation_dataset_creator = Doc2DocIRDatasetCreator(config)
    citation_dataset_creator.create_dataset(sub_datasets=False, kaggle=False, huggingface=True, save_reports=False)