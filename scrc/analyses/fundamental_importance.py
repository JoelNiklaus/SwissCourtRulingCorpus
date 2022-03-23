from typing import List, Union

import json
import nltk
from nltk import sent_tokenize, word_tokenize

nltk.download('punkt')

from root import ROOT_DIR
from scrc.preprocessors.abstract_preprocessor import AbstractPreprocessor
from scrc.utils.log_utils import get_logger
import pandas as pd
# TODO make abstract data base service or something to separate concerns better

from scrc.utils.main_utils import get_config, retrieve_from_cache_if_exists, save_df_to_cache, string_contains_one_of_list
import plotly.express as px

"""
Two large difficulties with this analysis:
1. Automatic sentence splitting in legal documents works very badly
2. Automatic negation detection is very hard in the multilingual setting and even harder in legal documents
"""


class FundamentalImportanceAnalysis(AbstractPreprocessor):

    def __init__(self, config: dict, negation_detection_type="xlmr"):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.negation_detection_type = negation_detection_type
        # Two methods: either search with these strings
        self.fundamental_importance_search_strings = {
            "de": ["Rechtsfrage von grundsätzlicher Bedeutung", "Frage von grundsätzlicher Bedeutung"],
            "fr": ["question juridique de principe", ],
            "it": ["questione di diritto di importanza fondamentale", "questione giuridica d''importanza fondamentale"],
        }
        # or search with law articles
        self.articles = {
            # We removed Art. 42 because it is being cited many times without relevance to fundamental importance
            # Zitate sehr abhängig vom Gerichtsschreiber
            "de": [
                "Art. 20 Abs. 2 BGG",
                # "Art. 42 Abs. 2 BGG", # causes too many false positives
                "Art. 74 Abs. 2 lit. a BGG",
                "Art. 83 Abs. 1 lit. f Ziff. 1 BGG", "Art. 83 lit. f Ziff. 1 BGG",
                "Art. 83 Abs. 1 lit. m BGG", "Art. 83 lit. m BGG",
                "Art. 83 Abs. 1 lit. w BGG", "Art. 83 lit. w BGG",
                "Art. 83 Abs. 1 lit. x BGG", "Art. 83 lit. x BGG",
                "Art. 84a BGG", "Art. 85 Abs. 2 BGG", "Art. 109 Abs. 1 BGG"
            ],
            "fr": [
                "art. 20 al. 2 LTF",
                # "art. 42 al. 2 LTF", # causes too many false positives
                "art. 74 al. 2 let. a LTF",
                "art. 83 al. 1 let. f n. 1 LTF", "art. 83 let. f n. 1 LTF",
                "art. 83 al. 1 let. m LTF", "art. 83 let. m LTF",
                "art. 83 al. 1 let. w LTF", "art. 83 let. w LTF",
                "art. 83 al. 1 let. x LTF", "art. 83 let. x LTF",
                "art. 84a LTF", "art. 85 al. 2 LTF", "art. 109 al. 1 LTF"
            ],
            "it": [
                "art. 20 cpv. 2 LTF",
                # "art. 42 cpv. 2 LTF", # causes too many false positives
                "art. 74 cpv. 2 lett. a LTF",
                "art. 83 cpv. 1 lett. f n. 1 LTF", "art. 83 lett. f n. 1 LTF",
                "art. 83 cpv. 1 lett. m LTF", "art. 83 lett. m LTF",
                "art. 83 cpv. 1 lett. w LTF", "art. 83 lett. w LTF",
                "art. 83 cpv. 1 lett. x LTF", "art. 83 lett. x LTF",
                "art. 84a LTF", "art. 85 cpv. 2 LTF", "art. 109 cpv. 1 LTF"
            ]
        }

    def retrieve_data(self, type, overwrite_cache):
        cache_file = self.analysis_dir / 'fundamental_importance.csv'

        engine = self.get_engine(self.db_scrc)
        # if cached just load it from there
        if not overwrite_cache:
            self.logger.info(f"Loading data from cache at {cache_file}")
            df = retrieve_from_cache_if_exists(cache_file)
            if not df.empty:
                return df

        # otherwise query it from the database
        df = pd.DataFrame()
        for lang in ["de", "fr", "it"]:
            # strpos is faster than LIKE, which is faster than ~ (regex search)
            where = f"spider = 'CH_BGer' AND "
            if type == "search_strings":
                search_strings = self.fundamental_importance_search_strings[lang]
            elif type == "articles":
                search_strings = self.articles[lang]
            else:
                raise ValueError("type should be either search_strings or articles")
            searches = [f"strpos(text, '{s}')>0" for s in search_strings]
            where += f"({' OR '.join(searches)})"
            columns = "language, chamber, date, html_url, paragraphs, text"
            df = df.append(next(self.select(engine, lang, columns=columns, where=where, chunksize=20000)))

        self.logger.info(f"Saving data to cache at {cache_file}")
        save_df_to_cache(df, cache_file)
        return df

    def analyze(self, type, overwrite_cache=False):
        self.logger.info(f"Analyzing Fundamental Importance by searching with {type}")
        self.analysis_dir = ROOT_DIR / f'analyses/fundamental_importance/{type}'
        self.analysis_dir.mkdir(parents=True, exist_ok=True)  # create folder if it does not exist yet

        df = self.retrieve_data(type, overwrite_cache=overwrite_cache)

        num_entries = df.groupby('language').text.count()
        num_entries.index = df.language.unique()
        self.logger.info(f"Found {num_entries.de} German decisions"
                         f", {num_entries.fr} French decisions "
                         f"and {num_entries.it} Italian decisions"
                         f" ({num_entries.sum()} in Total)")

        if self.negation_detection_type == "simple":
            self.logger.info("Splitting the text into sentences")
            df = df.apply(self.sentencize, axis="columns")

            self.logger.info("Removing the sentences that do not contain any legal question of fundamental importance")
            df = df.apply(self.filter_by_fundamental_importance, axis="columns")
            fundamental_importance_parts = "fundamental_importance_sentences"
        elif self.negation_detection_type == "xlmr":
            self.logger.info("Removing the paragraphs that do not contain any legal question of fundamental importance")
            df = df.apply(self.filter_by_fundamental_importance, axis="columns")
            fundamental_importance_parts = "fundamental_importance_paragraphs"

        self.logger.info("Removing the decisions where no direct mention of fundamental importance is found")
        df = df[df[fundamental_importance_parts].apply(lambda x: len(x)) > 0]
        self.logger.info(f"Removed {num_entries.sum() - df.text.count()} entries "
                         f"(Originally: {num_entries.sum()}, Now: {df.text.count()})")

        self.logger.info("Filtering decisions containing negations in the same sentence/paragraph "
                         "as the legal question of fundamental importance was found")
        df = df.apply(self.contains_negation_fundamental_importance, axis="columns")
        df['year'] = pd.to_datetime(df.date).dt.year
        df['legal_area'] = df.chamber.apply(get_legal_area)

        # TODO do smart aggregation (only negated if more than half is negated for example)
        self.logger.info("Counting the number of negated occurrences of fundamental importance")
        negated_mask = pd.array([bool(entry) for entry in df.negated.tolist()], dtype="boolean")
        negated_df = df[negated_mask]
        not_negated_df = df[~negated_mask]

        year_df = self.create_summary_df("year", df, negated_df, not_negated_df)
        print(year_df)

        legal_area_df = self.create_summary_df("legal_area", df, negated_df, not_negated_df)
        print(legal_area_df)

        lang_df = self.create_summary_df("language", df, negated_df, not_negated_df)
        print(lang_df)

        df.to_csv(self.analysis_dir / "fundamental_importance_result.csv")

        # Sample decisions to check the validity of the process
        df = df[["language", "negated", "legal_area", "html_url"]]

        sample = df.groupby(["language", "negated"]).sample(n=3, random_state=42)
        sample.to_csv(self.analysis_dir / "samples.csv")

    def create_summary_df(self, group_by, df, negated_df, not_negated_df):
        # create summary df for nice condensed presentation of results
        summary_df = pd.DataFrame()
        summary_df['negated'] = negated_df.groupby(group_by).text.count()
        summary_df['not_negated'] = not_negated_df.groupby(group_by).text.count()
        summary_df['total'] = df.groupby(group_by).text.count()
        summary_df = summary_df.fillna(0)
        total = summary_df.sum()
        total.name = 'All'  # add row for all
        # Assign sum of all rows of DataFrame as a new Row
        summary_df = summary_df.append(total.transpose())

        summary_df['not_negated_percentage'] = round(100 * summary_df.not_negated / summary_df.total, 2)
        row_names = df[group_by].unique().tolist()
        row_names.append("All")  # add name for last sum row
        summary_df.index = row_names  # give nice names to rows

        # draw histogram
        fig = px.bar(summary_df, x=summary_df.index, y='not_negated_percentage')
        fig.write_image(self.analysis_dir / f'{group_by}-histogram.png')

        summary_df.to_csv(self.analysis_dir / f'{group_by}.csv')

        return summary_df

    def contains_negation_fundamental_importance(self, df):
        # TODO bessere negation detection einbauen: https://spacy.io/universe/project/negspacy, https://drive.google.com/drive/folders/1md-_WBrg9x2Kp4g6jNExLJrEt5HBGL23
        all_negations = json.load((self.analysis_dir / 'negations.json').read_text())
        lang = df.language
        negations = all_negations[lang]["words"]
        if self.negation_detection_type == 'simple':
            df['fundamental_importance_tokens'] = self.nltk_word_tokenize(df.fundamental_importance_sentences, lang)
            df['negated'] = string_contains_one_of_list(df.fundamental_importance_tokens, negations)
        elif self.negation_detection_type == 'xlmr':

            # TODO paragraphen als Input nehmen und nicht ganzer Text
            # TODO negation cue listen beschaffen
            # TODO negation scope resolution mit modellen von anastassia machen
            # TODO aggregation pro vorkommnis des suchbegriffs machen (majority vote?)
            pass
        elif self.negation_detection_type == 'negex':
            raise ValueError("Negex is not implemented yet.")
        else:
            raise ValueError(f"Negation detection type {self.negation_detection_type} not supported")
        return df

    def filter_by_fundamental_importance(self, df):
        search_strings = self.fundamental_importance_search_strings[df.language]
        df['fundamental_importance_sentences'] = [sentence for sentence in df.sentences if
                                                  search_strings in sentence]
        df['fundamental_importance_paragraphs'] = [paragraph['text'] for paragraph in df.paragraphs if
                                                   search_strings in paragraph['text']]
        return df

    def sentencize(self, df):
        df['sentences'] = self.nltk_sentencize(df.text, df.language)
        return df

    def nltk_word_tokenize(self, text: Union[str, List], language: str) -> List[str]:
        if isinstance(text, List):
            text = " ".join(text)
        return self.nltk_tokenize(text, language, word_tokenize)

    def nltk_sentencize(self, text: str, language: str) -> List[str]:
        return self.nltk_tokenize(text, language, sent_tokenize)

    def nltk_tokenize(self, text: str, language: str, tokenization_func) -> List[str]:
        langs = {'de': 'german', 'fr': 'french', 'it': 'italian'}
        if language not in langs:
            raise ValueError(f"The language {language} is not supported.")
        return tokenization_func(text, language=langs[language])


if __name__ == '__main__':
    config = get_config()

    fundamental_importance_analysis = FundamentalImportanceAnalysis(config)
    fundamental_importance_analysis.analyze("search_strings", overwrite_cache=False)
    fundamental_importance_analysis.analyze("articles", overwrite_cache=False)
