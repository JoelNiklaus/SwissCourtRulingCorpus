from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.utils.log_utils import get_logger
import pandas as pd

from scrc.utils.main_utils import get_config


# TODO filter out cases where lower court is BVGer or a Handelsgericht because there BGer is only 2nd instance!
#  People go to 2nd instance much more often than to 3rd instance.
#  In many cases more than 50% of cases go to 2nd instance.
#  Probably less than 30% of cases go to 3rd instance. ==> Ask Daniel Kettiger
# TODO calculate this: Geschäftsberichte von grossen Gerichten holen und vergleichen mit Anzahl Bundesgerichten

class CriticalityDatasetCreator(DatasetCreator):
    """
    Creates a dataset with the text as input and whether it reaches the supreme court or not as labels
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.debug = False
        self.split_type = "date-stratified"
        self.dataset_name = "criticality_prediction"
        # TODO wait for section splitting in other courts for facts and considerations to be enabled
        self.feature_cols = ['text']  # ['facts', 'considerations', 'text']

    def get_dataset(self, feature_col, lang, save_reports):
        engine = self.get_engine(self.db_scrc)

        origin_chambers, supreme_court_df = self.query_supreme_court(engine, lang)

        df = pd.DataFrame()
        for origin_chamber in origin_chambers:
            origin_chamber_df = self.query_origin_chamber(feature_col, engine, lang, origin_chamber, supreme_court_df)
            df = df.append(origin_chamber_df)
        labels = ['non-critical', 'critical']

        return df, labels

    def query_origin_chamber(self, feature_col, engine, lang, origin_chamber, supreme_court_df):
        self.logger.info(f"Processing origin chamber {origin_chamber}")
        columns = ['id', 'chamber', 'date', 'extract(year from date) as year', feature_col]
        try:
            lower_court_df = next(self.select(engine, lang,
                                              columns=",".join(columns),
                                              where=f"chamber = '{origin_chamber}'",
                                              order_by="date",
                                              chunksize=self.get_chunksize()))
            lower_court_df = self.clean_df(lower_court_df, feature_col)
        except StopIteration:
            self.logger.error(f"No lower court rulings found for chamber {origin_chamber}. Returning empty dataframe.")
            return pd.DataFrame()
        # Include all decisions from the lower court with matching chamber and date: We have two error sources here:
        # 1. More than one decision at a given date in the lower court => too many decisions included
        # 2. Decision referenced from supreme court is not published in the lower court => not enough decisions included
        sc_origin_chamber_df = supreme_court_df[supreme_court_df.origin_chamber.str.fullmatch(origin_chamber)]
        date_match = lower_court_df.date.astype(str).isin(list(sc_origin_chamber_df.origin_date.astype(str)))
        critical_df = lower_court_df[date_match]
        critical_df['label'] = 'critical'
        non_critical_df = lower_court_df[~date_match]
        non_critical_df['label'] = 'non-critical'

        self.logger.info(f"# critical decisions: {len(critical_df.index)}")
        self.logger.info(f"# non-critical decisions: {len(non_critical_df.index)}")

        return critical_df.append(non_critical_df)

    def query_supreme_court(self, engine, lang):
        origin_chamber = "lower_court::json#>>'{chamber}' AS origin_chamber"
        origin_date = "lower_court::json#>>'{date}' AS origin_date"
        origin_file_number = "lower_court::json#>>'{file_number}' AS origin_file_number"
        try:
            supreme_court_df = next(self.select(engine, lang,
                                                columns=f"{origin_chamber}, {origin_date}, {origin_file_number}",
                                                where="court = 'CH_BGer'",
                                                order_by="origin_date",
                                                chunksize=self.get_chunksize()))
        except StopIteration:
            raise ValueError("No supreme court rulings found")
        supreme_court_df = supreme_court_df.dropna(subset=['origin_date', 'origin_chamber'])
        origin_chambers = list(supreme_court_df.origin_chamber.unique())
        self.logger.info(f"Found supreme court rulings with references to lower court rulings "
                         f"from chambers {origin_chambers}")
        return origin_chambers, supreme_court_df


if __name__ == '__main__':
    config = get_config()

    criticality_dataset_creator = CriticalityDatasetCreator(config)
    criticality_dataset_creator.create_dataset()
