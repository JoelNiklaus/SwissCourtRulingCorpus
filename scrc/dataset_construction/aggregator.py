import configparser
import gc
from os.path import exists
from pathlib import Path

import glob
from typing import Tuple

import pandas as pd

from root import ROOT_DIR
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.log_utils import get_logger


# import scrc.utils.monkey_patch  # prevent memory leak with pandas


class Aggregator(DatasetConstructorComponent):
    """
    Aggregates the files from the spiders into one containing all decisions and one containing all decisions for each language
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

    def combine_spiders(self, combine_raw=False) -> None:
        """build total df from individual spider dfs"""
        if combine_raw:
            self.logger.info(f"Started aggregating raw files")
            self.combine_spiders_by_base_path(self.raw_subdir)
            self.logger.info(f"Finished aggregating raw files")
        self.logger.info(f"Started aggregating clean files")
        self.combine_spiders_by_base_path(self.clean_subdir)
        self.logger.info(f"Finished aggregating clean files")

    def combine_spiders_by_base_path(self, base_path: Path) -> None:
        """build total df from individual spider dfs for a specific base path"""
        all_spiders_csv_path = self.get_all_spiders_csv_path(base_path)
        language_csv_paths = self.get_language_csv_paths(base_path, self.languages)
        combined_created, languages_created = self.perform_pre_checks(all_spiders_csv_path, language_csv_paths)
        if combined_created and languages_created:  # if we did both already
            self.logger.info(
                f"Both the combined csv and the language aggregated csv files exist already. "
                f"Terminating computation immediately. "
                f"In case you want to rerun the aggregations, please delete the aggregated files.")
            return  # stop immediately

        spider_list = [Path(spider).stem for spider in glob.glob(f"{str(base_path)}/*.parquet")]
        for spider in spider_list:
            df = pd.read_parquet(base_path / (spider + '.parquet'))  # read df of spider

            if not combined_created:  # if we still need create the combined file
                self.logger.info(f"Adding spider {spider} to combined file")
                self.append_spider_to_agg_file(df, all_spiders_csv_path)  # add full df to file containing everything

            if not languages_created:  # if we still need create the language aggregated files
                self.logger.info(f"Adding spider {spider} to language aggregated files")
                for lang in self.languages:
                    lang_df = df[df['language'].str.contains(lang, na=False)]  # select only decisions by language
                    self.append_spider_to_agg_file(lang_df, language_csv_paths[lang])

            gc.collect()  # should prevent memory leak

    def perform_pre_checks(self, all_spiders_csv_path: Path, language_csv_paths: dict) -> Tuple[bool, bool]:
        """Check if we did these aggregations already before"""
        if all_spiders_csv_path.exists():  # Delete the combined file in order to recreate it.
            self.logger.info(f"Combined csv file already exists at {all_spiders_csv_path}")
            combined_already_created = True
        else:
            self.logger.info(f"Combining all individual spider dataframes and saving to {all_spiders_csv_path.name}")
            combined_already_created = False
        # Delete at least one of the language aggregated files in order to recreate them
        if all(map(exists, language_csv_paths.values())):
            self.logger.info(f"Language aggregated csv files already exist")
            languages_already_created = True
        else:
            self.logger.info("Splitting individual spider dataframes by language and saving to _{lang}.csv")
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
