import itertools
from collections import Counter
import datasets
from datasets import load_dataset
from datasets import load_dataset_builder
import json
import multiprocessing
from multiprocessing import Pool
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

from scrc.data_classes.law_citation import LawCitation
from scrc.data_classes.ruling_citation import RulingCitation
from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.enums.section import Section
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
            contains only the train split of queries.jsonl (years 2000 - 2015)
        val.jsonl:
            contains only the validation split of queries.jsonl (years 2016 - 2017)
        test.jsonl:
            contains only the test split of queries.jsonl (years 2018 - 2022)

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

    def __init__(self, config: dict, debug: bool = False):
        super().__init__(config, debug)
        self.logger = get_logger(__name__)

        self.split_type = "date-stratified"
        self.dataset_name = "doc2doc_ir"
        self.feature_cols = [Section.FACTS, Section.CONSIDERATIONS, Section.RULINGS]
        self.labels = ['laws', 'cited_rulings']

        self.num_ruling_citations = 1000  # the 1000 most common ruling citations will be included

        self.ruling_df = self.load_rulings()
        self.available_bges = self.ruling_df.text.tolist()
        self.load_law_articles()

        pandarallel.initialize(progress_bar=True)
        tqdm.pandas()

    def load_law_articles(self):
        self.logger.info(f"Loading reference law articles")
        # todo find solution to load directly from huggingface
        # laws = load_dataset("rcds/swiss_legislation")
        laws = pd.read_json(self.corpora_subdir / "lexfind.jsonl", lines=True)
        laws = laws[laws.canton.str.contains("ch")]  # only federal laws so far
        laws = laws[laws.abbreviation.str.len() > 1]  # only keep the ones with an abbreviation
        laws.abbreviation = laws.abbreviation.str.strip()
        # TODO think about if we need to combine the laws in the different languages to one single law with representations in different languages
        # law articles exist for german, french and italian but they do have different uuid but the same sr_number! (abbreviations not always the same)
        self.law_abbrs = laws[["language", "abbreviation", "sr_number", "uuid"]]
        self.available_laws = set(laws.abbreviation.unique().tolist())
        self.logger.info(f"Found {len(self.available_laws)} laws")
        # This can take quite a long time
        laws_path = self.get_dataset_folder() / "laws.jsonl"
        #if not laws_path.exists():
         #   self.logger.info("Extracting individual articles from laws")
          #  articles = pd.concat([self.extract_article_info(row) for _, row in laws.iterrows()], ignore_index=True)
           # articles.to_json(laws_path, orient="records", lines=True)

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

    def prepare_dataset(self, save_reports, court_string):
        data_to_load = {
            "section": True, "file": True, "file_number": True,
            "judgment": False, "citation": True, "lower_court": True,
            "law_area": True, "law_sub_area": True
        }

        df = self.get_df(self.get_engine(self.db_scrc), data_to_load)
        df.dropna(subset=['citations'], inplace=True)

        # df = self.process_citation(df)

        def parallelize_dataframe(df, func, n_cores=10):
            df_split = np.array_split(df, n_cores)
            pool = Pool(n_cores)
            df = pd.concat(pool.map(func, df_split))
            pool.close()
            pool.join()
            return df

        df = parallelize_dataframe(df, self.process_citation_1)
        df = parallelize_dataframe(df, self.process_citation_2)

        df = df.apply(self.mask_citations, axis="columns")

        # df = self.do_some_fancy_stuff(df, cit_type)

        df.drop(['laws_citations', 'rulings_citations'], axis=1, inplace=True)

        # TODO extract ruling citations from other decisions and test the extraction regexes on the CH_BGer

        rulings_labels, _ = list(np.unique(np.hstack(self.ruling_df.text), return_index=True))
        laws_labels, _ = list(np.unique(np.hstack(self.available_laws), return_index=True))
        label_list = [laws_labels, rulings_labels]
        dataset = datasets.Dataset.from_pandas(df)
        return dataset, label_list


    def mask_citations(self, row, law_mask_token="<ref-law>", ruling_mask_token="<ref-ruling>"):
        """
        Mask citations so that they don't help in finding the right document,
        but it should only use the context
        """
        citations = row['citations']
        for feature_col in self.get_feature_col_names():
            text = row[feature_col]
            if text is not None and isinstance(text, str) and text != []:
                for citation in citations:
                    if citation['name'] == 'ruling':
                        text = text.replace(citation['text'], ruling_mask_token)
                    elif citation['name'] == 'law':
                        text = str(text).replace(citation['text'], law_mask_token)
            row[feature_col] = text
        return row

    def process_citation_1(self, df):
        # creating processes for each of the functions
        self.logger.info(f"Processing the citations.")
        # get an array of the rulings file_number that were cited
        df['rulings_citations'] = df.citations.apply(self.get_citation, type='ruling')
        # get the law article for each law citation
        df['laws_citations'] = df.citations.apply(self.get_citation, type='law')
        return df

    def process_citation_2(self, df):
        # now find for each bge file_number the decision_id
        self.logger.info("getting bge decision Ids")
        df['cited_rulings'] = df.rulings_citations.apply(self.get_decision_ids_of_bges)
        # find uuids for cited laws in all languages
        self.logger.info("getting law uuids")
        df['laws'] = df.laws_citations.apply(self.get_uuid_of_laws)
        return df

    def get_uuid_of_laws(self, law_cits):
        self.counter = self.counter + 1
        if int(self.counter) % 10000 == 0:
            self.logger.info("Processed another 10'000 citations")
        if law_cits is None:
            return []
        uuids = []
        for item in law_cits:
            sr_number = item['law']
            laws = self.law_abbrs[(self.law_abbrs.sr_number.str.strip() == sr_number)]  # cannot differ French and Italian
            if len(laws.index) == 0:
                # only include citations that we can find in our corpus
                raise ValueError(f"The sr_number ({sr_number}) cannot be found.")
            abbreviations = []
            uuids = []
            for index, row in laws.iterrows():
                abbreviations.append(row.abbreviation)
                uuids.append(row.uuid)
        return uuids

    def get_decision_ids_of_bges(self, bge_cits):
        self.counter = self.counter + 1
        if int(self.counter) % 10000 == 0:
            self.logger.info("Processed another 10'000 citations")
        if bge_cits is None or bge_cits == []:
            return []
        ids = []
        for bge_file_name in bge_cits:
            # find decision_id in ruling_df
            decision_id = self.find_decision_id_for_ruling(bge_file_name)
            if decision_id is not None:
                ids.append(decision_id)
        return ids

    def find_decision_id_for_ruling(self, bge_file_name):
        match = self.ruling_df['text'] == str(bge_file_name)
        if len(self.ruling_df[match]) > 0:
            return self.ruling_df[match, 'decision_id'].iloc[:1]
        return None

    def get_law_citation(self, citations_text):
        # create law for citations
        law = LawCitation(citations_text, self.law_abbrs)
        return {'text': citations_text, 'law': law.sr_number}

    def do_some_fancy_stuff(self, df, cit_type):
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
        figure_path = self.get_dataset_folder() / f"{name}_citations.png"
        csv_path = self.get_dataset_folder() / f"{name}_citations.csv"

        citations = pd.DataFrame.from_dict(citation_frequencies, orient="index", columns=["frequency"])
        citations = citations.sort_values(by=['frequency'], ascending=False)
        citations.to_csv(csv_path)

        ax = citations[:top_n].plot.bar(use_index=True, y='frequency', rot=90)
        ax.get_figure().savefig(figure_path, bbox_inches="tight")

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

    def get_df_from_json(self):
        file_path = self.get_dataset_folder() / "df.jsonl"
        with open(file_path, "r") as json_file:
            json_list = list(json_file)
        a_list = []
        for json_str in json_list:
            result = json.loads(json_str)
            a_list.append(result)
            assert isinstance(result, dict)
        df = pd.DataFrame.from_records(a_list)
        self.logger.info(len(df))
        self.logger.info(df.columns)
        return df


if __name__ == '__main__':
    config = get_config()
    citation_dataset_creator = Doc2DocIRDatasetCreator(config)
    citation_dataset_creator.create_dataset(sub_datasets=False, kaggle=False, save_reports=False)
