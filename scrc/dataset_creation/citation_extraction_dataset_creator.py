import itertools
from collections import Counter
import datasets
import json
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

from scrc.data_classes.law_citation import LawCitation
from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.enums.section import Section
from scrc.utils.log_utils import get_logger
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from datasets import load_dataset
import re

from scrc.utils.main_utils import get_config
from scrc.dataset_creation.report_creator import plot_input_length, plot_cit_amounts


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

class CitationExtractionDatasetCreator(DatasetCreator):
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

    # usefulness of law articles still unclear. ALCP, OPB, VEP, LMV, SBG, LSV
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
        self.dataset_name = "citation_extraction"
        self.feature_cols = [Section.CONSIDERATIONS]
        self.filter_cols = [Section.CONSIDERATIONS]
        self.num_ruling_citations = 1000  # the 1000 most common ruling citations will be included
        self.metadata = ['year', 'chamber', 'region', 'origin_chamber', 'origin_court', 'origin_canton',
                         'law_area', 'NER_labels']
        self.reports_folder = self.create_dir(self.get_dataset_folder(), f'CH_BGer/reports')
        self.ruling_df, self.available_rulings_dict = self.load_rulings()
        self.load_law_articles()

        pandarallel.initialize(progress_bar=True)
        tqdm.pandas()

    def load_law_articles(self):
        """
        Load all law articles and save them
        """
        self.logger.info(f"Loading reference law articles")
        law_dataset = load_dataset('rcds/swiss_legislation', split='train')
        law_dataset = law_dataset.filter(lambda row: row['canton'] == 'ch')
        laws = pd.DataFrame(law_dataset)
        laws = laws[laws.abbreviation.str.len() > 1]  # only keep the ones with an abbreviation
        laws.abbreviation = laws.abbreviation.str.strip()
        # TODO think about if we need to combine the laws in the different languages to one single law with representations in different languages
        #  law articles exist for german, french and italian but they do have different uuid but the same sr_number! (abbreviations not always the same)
        self.law_abbrs = laws[["language", "abbreviation", "sr_number", "uuid", 'pdf_content']]
        self.available_laws = set(laws.abbreviation.unique().tolist())
        self.logger.info(f"Found {len(self.available_laws)} laws")
        print(self.available_laws)

    def prepare_dataset(self, save_reports, court_string):
        """
        get of all bger cases and set criticality labels: bge_label and citation_label
        :param save_reports:    whether or not to compute and save reports # TODO is False as default due to a bug
        :param court_string:    specifies court, in this case it is CH_BGer
        :return:                dataframe and list of labels
        """
        data_to_load = {
            "section": True, "file": True, "file_number": True,
            "judgment": False, "citation": True, "lower_court": True,
            "law_area": True, "law_sub_area": True
        }

        df = self.get_df(self.get_engine(self.db_scrc), data_to_load, use_cache=False)

        # get rid of all cases where no citations were found
        df.dropna(subset=['citations'], inplace=True)

        # cut process into two pieces to make sure it is handled sequentially
        df = self.parallelize_dataframe(df, self.process_citation_1)
        df = self.parallelize_dataframe(df, self.process_citation_2)
        self.logger.info("finnished processing df")

        plot_cit_amounts(self.reports_folder, df)

        # df = df.apply(self.mask_citations, axis="columns")
        df = self.truncate_considerations(df)

        df = self.create_NER_labels(df)

        # df = self.do_some_fancy_stuff(df, cit_type)

        df.drop(['laws_citations', 'rulings_citations'], axis=1, inplace=True)
        df['laws'] = df['laws'].astype(str)
        df['cited_rulings'] = df['cited_rulings'].astype(str)

        # TODO extract ruling citations from other decisions and test the extraction regexes on the CH_BGer

        for feature_col in self.get_feature_col_names():
            tmp_df = df.rename(columns={f'{feature_col}_num_tokens_bert': "num_tokens_bert", f'{feature_col}_num_tokens_spacy': "num_tokens_spacy"})
            plot_input_length(self.reports_folder, tmp_df, feature_col)

        rulings_labels, _ = list(np.unique(np.hstack(df.cited_rulings), return_index=True))
        laws_labels, _ = list(np.unique(np.hstack(df.laws), return_index=True))
        label_list = [laws_labels, rulings_labels]

        dataset = datasets.Dataset.from_pandas(df)
        return dataset, label_list

    def mask_citations(self, row, law_mask_token="<ref-law>", ruling_mask_token="<ref-ruling>"):
        """
        Mask citations, so they don't help in finding the right document, it should only use the context
        :param: row:    a row in the dataframe
        :return:        row with masked citations for all feature_cols
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
        """
        First part of processing citations, extracts for each citation (ruling and law) the needed information
        :param: df:     dataframe containing all bger
        """
        self.logger.info(f"Processing the citations.")
        # get an array of the rulings file_number that were cited
        df['rulings_citations'] = df.citations.apply(self.get_citation, type='ruling')
        # get the law article for each law citation
        df['laws_citations'] = df.citations.apply(self.get_citation, type='law')
        return df

    def process_citation_2(self, df):
        """
        Second part of processing citations, finds a unique representation for each law and ruling citations
        :param: df:     dataframe containing all bger
        """
        # find for each bge file_number the decision_id
        df['cited_rulings'] = df.rulings_citations.apply(self.get_decision_ids_of_bges)
        df['cited_rulings_count'] = df['cited_rulings'].apply(lambda x: len(x))
        self.logger.info(f"There were {self.counter} bge cits without actual bge.")
        self.counter = 0
        # find uuids for cited laws in all languages
        df['laws'] = df.laws_citations.apply(self.get_uuid_of_laws)
        df['laws_count'] = df['laws'].apply(lambda x: len(x))
        match = df['cited_rulings_count'] == 0
        print(f"There are {len(df[match].index)} cases without cited_rulings.")
        match = df['laws_count'] == 0
        print(f"There are {len(df[match].index)} cases without laws.")
        self.logger.info(f"There were {self.counter} found laws in multiple languages.")
        return df

    def get_uuid_of_laws(self, law_cits):
        """
        Find for a given law its uuid. Each law is written in multiple languages, so we get multiple uuids per law.
        """
        if law_cits is None:
            return []
        uuids = []
        for item in law_cits:
            sr_number = item['law']
            laws = self.law_abbrs[(self.law_abbrs.sr_number.str.strip() == sr_number)]  # cannot differ French and Italian
            if len(laws.index) == 0:
                # only include citations that we can find in our corpus
                raise ValueError(f"The sr_number ({sr_number}) cannot be found.")
            for index, row in laws.iterrows():
                uuids.append(row.uuid)
                self.counter = self.counter + 1
        uuids = list(set(uuids))
        return uuids

    def get_decision_ids_of_bges(self, bge_cits):
        """
        Find decision_id for ruling_citation by given bge_file_number
        """
        if bge_cits is None or bge_cits == []:
            return []
        ids = []
        for bge_file_name in bge_cits:
            if bge_file_name is not None:
                # find decision_id in ruling_df
                decision_id = self.find_decision_id_for_ruling(bge_file_name)
                if decision_id is not None:
                    ids.append(decision_id)
        return ids

    def find_decision_id_for_ruling(self, bge_file_name):
        """
        This might not be possible for all citations because the citation does
        sometimes cite a specific page instead of the beginning of the ruling, so bge_file_name in invalid
        """
        bge_file_name = bge_file_name.replace('-', '_')
        bge_file_name = bge_file_name.replace(' ', '_')
        match = self.ruling_df['file_number'] == str(bge_file_name)
        if len(self.ruling_df[match]) > 0:
            # todo add ruling to qrels dict
            return str(self.ruling_df[match].iloc[0].loc['decision_id'])
        return None

    def get_law_citation(self, citations_text):
        """
        create a law object for each given citation_text, to extract the needed information: sr_number
        This number is unique for each law, so we can identify the needed article by the sr_number in a next step
        """
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

    def create_NER_labels(self, df):
        """
        1. whitespace split the 'considerations' column to get a list of words
        2. create a new column 'NER_labels' with the same length as the 'considerations' column and initialize it with 'O'
        3. set the NER labels for the citations
        6. return the df
        :param df:
        :return:
        """

        # split by whitespace and also special chars like .,;:() etc. as separate words
        df['considerations'] = df['considerations'].apply(lambda x: re.findall(r"[\w]+|[^\s\w]", x))
        df['NER_labels'] = df['considerations'].apply(lambda x: ['O'] * len(x))
        df['NER_labels'] = df.apply(lambda x: self.set_NER_labels(x['NER_labels'], x['citations'], x['considerations']), axis=1)
        return df

    def set_NER_labels(self, NER_labels, citations, considerations):
        words = considerations
        num_found = 0
        for citation in citations:
            if citation['name'] == 'law':
                begin_label, end_label = 'B-LAW', 'I-LAW'
            else:
                begin_label, end_label = 'B-CITATION', 'I-CITATION'
            # split using same regex as in create_NER_labels
            citation_words = re.findall(r"[\w]+|[^\s\w]", citation['text'])
            for i in range(len(words)):
                if words[i:i + len(citation_words)] == citation_words:
                    num_found += 1
                    NER_labels[i] = begin_label
                    for j in range(i + 1, i + len(citation_words)):
                        if j < len(NER_labels):
                            NER_labels[j] = end_label
        return NER_labels

    def truncate_considerations(self, df):
        # split the considerations into paragraphs and only keep so many paragraphs that the total number of words is <= 215
        for i, row in df.iterrows():
            paragraphs = row['considerations'].split('\n')
            num_words = 0
            truncated_paragraphs = []
            for paragraph in paragraphs:
                words_in_paragraph = len(paragraph.split(' '))
                if num_words + words_in_paragraph <= 215:
                    num_words += words_in_paragraph
                    truncated_paragraphs.append(paragraph)
                else:
                    break
            df.at[i, 'considerations'] = '\n'.join(truncated_paragraphs)
        return df


if __name__ == '__main__':
    config = get_config()
    citation_extraction_dataset_creator = CitationExtractionDatasetCreator(config, debug=False)
    citation_extraction_dataset_creator.create_dataset(sub_datasets=False, kaggle=False, save_reports=True)

    # python -m scrc.dataset_creation.citation_extraction_dataset_creator