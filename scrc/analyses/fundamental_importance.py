from typing import List, Union

import nltk
from nltk import sent_tokenize, word_tokenize

nltk.download('punkt')

from root import ROOT_DIR
from scrc.preprocessors.abstract_preprocessor import AbstractPreprocessor
from scrc.utils.log_utils import get_logger
import pandas as pd
# TODO make abstract data base service or something to separate concerns better
from scrc.utils.main_utils import get_config, string_contains_one_of_list


class FundamentalImportanceAnalysis(AbstractPreprocessor):

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.fundamental_importance_search_strings = {
            "de": "Rechtsfrage von grundsätzlicher Bedeutung",
            "fr": "question juridique de principe",
            "it": "questione di diritto di importanza fondamentale",
        }

    def retrieve_data(self, overwrite_cache=False):
        cache_file = ROOT_DIR / 'scrc/analyses/fundamental_importance.csv'
        engine = self.get_engine(self.db_scrc)
        # if cached just load it from there
        if cache_file.exists() and not overwrite_cache:
            self.logger.info(f"Loading data from cache at {cache_file}")
            df = pd.read_csv(cache_file)
            return df

        # otherwise query it from the database
        df = pd.DataFrame()
        for lang in ["de", "fr", "it"]:
            # strpos is faster than LIKE, which is faster than ~ (regex search)
            where = f"spider = 'CH_BGer' AND strpos(text, '{self.fundamental_importance_search_strings[lang]}')>0"
            columns = "language, chamber, date, text, html_url"
            df = df.append(next(self.select(engine, lang, columns=columns, where=where, chunksize=5000)))

        self.logger.info(f"Saving data to cache at {cache_file}")
        df.to_csv(cache_file, index=False)  # save cache file for next time
        return df

    def analyze(self):
        df = self.retrieve_data(overwrite_cache=False)

        self.logger.info("Splitting the text into sentences")
        df = df.apply(self.sentencize, axis="columns")

        self.logger.info("Removing the sentences that do not contain any legal question of fundamental importance")
        df = df.apply(self.filter_by_fundamental_importance, axis="columns")
        # print(df.sentences.str.len()) # print the number of sentences
        # print(df.fundamental_importance_sentences.str.len()) # print the number of sentences

        self.logger.info("Filtering decisions containing negations in the same sentence "
                         "as the legal question of fundamental importance was found")
        df = df.apply(self.contains_negation_fundamental_importance, axis="columns")

        self.logger.info("Counting the number of negated sentences")
        negated_mask = pd.array([bool(entry) for entry in df.negated.tolist()], dtype="boolean")
        negated_df = df[negated_mask]
        not_negated_df = df[~negated_mask]
        print(negated_df.groupby("language").text.count())
        print(not_negated_df.groupby("language").text.count())
        print(df.groupby("language").text.count())

        summary_df = pd.DataFrame()
        summary_df['negated'] = negated_df.groupby("language").text.count()
        summary_df['not_negated'] = not_negated_df.groupby("language").text.count()
        summary_df['total'] = df.groupby("language").text.count()
        summary_df['not_negated_percentage'] = round(100 * summary_df.not_negated / summary_df.total, 2)
        summary_df.index = ['de', 'fr', 'it']  # give nice names to rows

        print(summary_df)

        # use this for debugging large dfs so it can show everything
        # with pd.option_context('display.max_rows', None, 'display.max_columns',None):
        # print(df)

    def contains_negation_fundamental_importance(self, df):

        negation_fundamental_importance_search_strings = {
            "de": ["keine", "nicht", "mangels", ],
            "fr": ["aucune", "ne", "pas", "absence", ],
            "it": ["non", "né", ]
        }
        df['fundamental_importance_tokens'] = self.nltk_word_tokenize(df.fundamental_importance_sentences, df.language)
        df['negated'] = string_contains_one_of_list(df.fundamental_importance_tokens,
                                                    negation_fundamental_importance_search_strings[
                                                        df.language])
        return df

    def filter_by_fundamental_importance(self, df):
        df['fundamental_importance_sentences'] = [sentence for sentence in df.sentences if
                                                  self.fundamental_importance_search_strings[df.language] in sentence]
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
    fundamental_importance_analysis.analyze()
