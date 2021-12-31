from root import ROOT_DIR
from scrc.dataset_creation.judgment_dataset_creator import convert_to_binary_judgments
from scrc.preprocessors.abstract_preprocessor import AbstractPreprocessor
from scrc.utils.log_utils import get_logger
import pandas as pd
# TODO make abstract data base service or something to separate concerns better
from scrc.utils.main_utils import get_config


class BuildingViolationsAnalysis(AbstractPreprocessor):

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.ARE_names = {
            "de": "Bundesamt für Raumentwicklung",
            "fr": "Office fédéral du développement territorial",
            "it": "Ufficio federale dello sviluppo territoriale",
        }
        self.law_abbrs = {"de": "RPG", "fr": "LAT", "it": "LPT"}

    def retrieve_data(self, overwrite_cache=False):
        cache_file = ROOT_DIR / 'scrc/analyses/building_violations.csv'
        engine = self.get_engine(self.db_scrc)
        # if cached just load it from there
        if cache_file.exists() and not overwrite_cache:
            self.logger.info(f"Loading data from cache at {cache_file}")
            df = pd.read_csv(cache_file)
            df.parties = df.parties.apply(lambda x: eval(x))  # parse dict string to dict again
            df.original_judgments = df.original_judgments.apply(lambda x: eval(x))  # parse list string to list again
            return df

        # otherwise query it from the database
        df = pd.DataFrame()
        for lang in ["de", "fr", "it"]:
            where = f"spider = 'CH_BGer' AND considerations ~ '[Aa]rt\\. 24[bcde].{{1,20}}{self.law_abbrs[lang]}'"
            columns = "language, chamber, date, html_url, parties, judgments"
            df = df.append(next(self.select(engine, lang, columns=columns, where=where, chunksize=200)))

        # Do some preprocessing
        df = df.dropna(subset=["parties", "judgments"])  # remove decisions with no parties and no judgments
        original_judgments = df.judgments.copy()
        df = convert_to_binary_judgments(df)  # clean judgments so that we get one clear outcome
        df['original_judgments'] = original_judgments  # keep original judgments for later analysis
        # df.dropna(subset=["original_judgments", "judgments"])

        self.logger.info(f"Saving data to cache at {cache_file}")
        df.to_csv(cache_file, index=False)  # save cache file for next time
        return df

    def analyze(self):
        df = self.retrieve_data(overwrite_cache=False)

        ARE_is_plaintiff_mask = df.apply(self.filter_parties, args=(True,), axis="columns")
        ARE_is_defendant_mask = df.apply(self.filter_parties, args=(False,), axis="columns")
        ARE_is_plaintiff = df[ARE_is_plaintiff_mask]
        ARE_is_defendant = df[ARE_is_defendant_mask]
        non_ARE_is_plaintiff = df[~ARE_is_plaintiff_mask]
        non_ARE_is_defendant = df[~ARE_is_defendant_mask]

        def get_approval_count(df):
            value_counts_dict = df.judgments.value_counts().to_dict()
            if 'approval' in value_counts_dict:
                return value_counts_dict['approval']
            else:
                return 0

        dfs = [df, ARE_is_plaintiff, non_ARE_is_plaintiff, ARE_is_defendant, non_ARE_is_defendant]
        summary_df = pd.DataFrame()
        summary_df['approvals'] = [get_approval_count(tmp_df) for tmp_df in dfs]
        summary_df['total'] = [len(tmp_df.index) for tmp_df in dfs]
        # give nice names to rows
        summary_df.index = ['all cases', 'ARE is plaintiff', 'non-ARE is plaintiff', 'ARE is defendant',
                            'non-ARE is defendant']
        summary_df['approval_percentage'] = round(100 * summary_df.approvals / summary_df.total, 2)
        print(summary_df)

    def filter_parties(self, df, is_ARE_plaintiff=True):
        parties_number = '0' if is_ARE_plaintiff else '1'
        return df.parties[parties_number]['party'][0]['name'] == self.ARE_names[df.language]


if __name__ == '__main__':
    config = get_config()

    building_violations_analysis = BuildingViolationsAnalysis(config)
    building_violations_analysis.analyze()
