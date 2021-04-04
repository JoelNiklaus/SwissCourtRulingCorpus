import configparser
from os.path import exists
from pathlib import Path

import glob
from typing import Tuple

import pandas as pd

from root import ROOT_DIR
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.log_utils import get_logger

logger = get_logger(__name__)


class Aggregator(DatasetConstructorComponent):
    """
    Extracts the textual and meta information from the court rulings files and saves it in csv files for each spider
    and in one for all spiders combined
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.languages = ['de', 'fr', 'it']

    def combine_spiders(self) -> None:
        """build total df from individual spider dfs"""
        logger.info(f"Started aggregating raw csv files")
        self.combine_spiders_by_base_path(self.raw_csv_subdir)
        logger.info(f"Finished aggregating raw csv files")
        logger.info(f"Started aggregating clean csv files")
        self.combine_spiders_by_base_path(self.clean_csv_subdir)
        logger.info(f"Finished aggregating clean csv files")

    def combine_spiders_by_base_path(self, base_path: Path) -> None:
        """build total df from individual spider dfs for a specific base path"""
        all_spiders_csv_path = self.get_all_spiders_csv_path(base_path)
        language_csv_paths = self.get_language_csv_paths(base_path, self.languages)
        combined_created, languages_created = self.perform_pre_checks(all_spiders_csv_path, language_csv_paths)
        if combined_created and languages_created:  # if we did both already
            logger.info(
                f"Both the combined csv and the language aggregated csv files exist already. "
                f"Terminating computation immediately. "
                f"In case you want to rerun the aggregations, please delete the aggregated files.")
            return  # stop immediately

        spider_list = [Path(spider).stem for spider in glob.glob(f"{str(base_path)}/*")]
        for spider in spider_list:
            df = pd.read_csv(base_path / (spider + '.csv'))  # read df of spider

            if not combined_created:  # if we still need create the combined file
                logger.info(f"Adding spider {spider} to combined file")
                self.append_spider_to_agg_file(df, all_spiders_csv_path)  # add full df to file containing everything

            if not languages_created:  # if we still need create the language aggregated files
                logger.info(f"Adding spider {spider} to language aggregated files")
                for lang in self.languages:
                    lang_df = df[df['language'].str.contains(lang, na=False)]  # select only decisions by language
                    self.append_spider_to_agg_file(lang_df, language_csv_paths[lang])

    @staticmethod
    def perform_pre_checks(all_spiders_csv_path: Path, language_csv_paths: dict) -> Tuple[bool, bool]:
        """Check if we did these aggregations already before"""
        if all_spiders_csv_path.exists():  # Delete the combined file in order to recreate it.
            logger.info(f"Combined csv file already exists at {all_spiders_csv_path}")
            combined_already_created = True
        else:
            logger.info(f"Combining all individual spider dataframes and saving to {all_spiders_csv_path.name}")
            combined_already_created = False
        # Delete at least one of the language aggregated files in order to recreate them
        if all(map(exists, language_csv_paths.values())):
            logger.info(f"Language aggregated csv files already exist")
            languages_already_created = True
        else:
            logger.info("Splitting individual spider dataframes by language and saving to _{lang}.csv")
            languages_already_created = False
        return combined_already_created, languages_already_created

    @staticmethod
    def get_language_csv_paths(base_path: Path, languages: list) -> dict:
        """get the language specific aggregate path"""
        return {language: base_path / f'_{language}.csv' for language in languages}

    @staticmethod
    def get_all_spiders_csv_path(base_path: Path) -> Path:
        """get the total aggregate path"""
        return base_path / '_all.csv'

    @staticmethod
    def append_spider_to_agg_file(df: pd.DataFrame, path: Path) -> None:
        """Appends a given df to a given aggregated csv file"""
        mode, header = 'a', False  # normally append and don't add column names
        if not path.exists():  # if the combined file does not exist yet
            mode, header = 'w', True  # => normal write mode and add column names
        df.to_csv(path, mode=mode, header=header, index=False)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    aggregator = Aggregator(config)
    aggregator.combine_spiders()