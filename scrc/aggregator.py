import configparser
from os.path import exists
from pathlib import Path

import glob

import pandas as pd
from tqdm import tqdm

from root import ROOT_DIR
from scrc.utils.log_utils import get_logger

logger = get_logger(__name__)


class Aggregator:
    """
    Extracts the textual and meta information from the court rulings files and saves it in csv files for each court
    and in one for all courts combined
    """

    def __init__(self, config: dict):
        self.data_dir = ROOT_DIR / config['dir']['data_dir']
        self.courts_dir = self.data_dir / config['dir']['courts_subdir']
        self.csv_dir = self.data_dir / config['dir']['csv_subdir']
        self.raw_csv_subdir = self.csv_dir / config['dir']['raw_csv_subdir']
        self.clean_csv_subdir = self.csv_dir / config['dir']['clean_csv_subdir']
        self.all_courts_csv_path = self.csv_dir / '_all.csv'
        self.languages = ['de', 'fr', 'it']
        self.language_csv_paths = {language: self.csv_dir / f'_{language}.csv' for language in self.languages}

    def combine_courts(self) -> None:
        """build total df from individual court dfs"""
        court_list = [Path(court).stem for court in glob.glob(f"{str(self.clean_csv_subdir)}/*")]

        combined_already_created, languages_already_created = False, False
        if self.all_courts_csv_path.exists():  # Delete the combined file in order to recreate it.
            logger.info(f"Combined csv file already exists at {self.all_courts_csv_path}")
            combined_already_created = True
        else:
            logger.info(f"Combining all individual court dataframes and saving to {self.all_courts_csv_path.name}")

        # Delete at least one of the language aggregated files in order to recreate them
        if all(map(exists, self.language_csv_paths.values())):
            logger.info(f"Language aggregated csv files already exist")
            languages_already_created = True
        else:
            logger.info("Splitting individual court dataframes by language and saving to _{lang}.csv")

        for court in court_list:
            df = pd.read_csv(self.clean_csv_subdir / (court + '.csv'))  # read df of court

            if not combined_already_created:  # if we still need create the combined file
                logger.info(f"Adding court {court} to combined file")
                self.append_court_to_agg_file(df, self.all_courts_csv_path)  # add full df to file containing everything

            if not languages_already_created:  # if we still need create the language aggregated files
                logger.info(f"Adding court {court} to language aggregated files")
                for lang in self.languages:
                    lang_df = df[df['language'].str.contains(lang, na=False)]  # select only decisions by language
                    self.append_court_to_agg_file(lang_df, self.language_csv_paths[lang])

    @staticmethod
    def append_court_to_agg_file(df, path) -> None:
        """Appends a given df to a given aggregated csv file"""
        mode, header = 'a', False  # normally append and don't add column names
        if not path.exists():  # if the combined file does not exist yet
            mode, header = 'w', True  # => normal write mode and add column names
        df.to_csv(path, mode=mode, header=header, index=False)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    aggregator = Aggregator(config)
    aggregator.combine_courts()
