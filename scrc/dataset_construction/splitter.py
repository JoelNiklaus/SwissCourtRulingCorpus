import configparser
import gc
from pathlib import Path

import glob

import pandas as pd

from root import ROOT_DIR
from scrc.dataset_construction.dataset_constructor_component import DatasetConstructorComponent
from scrc.utils.decorators import slack_alert
from scrc.utils.log_utils import get_logger

import scrc.utils.monkey_patch  # prevent memory leak with pandas


class Splitter(DatasetConstructorComponent):
    """
    Splits the spider files into files by each chamber and stores them hiearchically by language => canton => court => chamber
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

    def split_spiders(self, ) -> None:
        self.logger.info(f"Started splitting clean files")
        self.split_spiders_by_base_path(self.clean_subdir)
        self.logger.info(f"Finished splitting clean files")

    def split_spiders_by_base_path(self, base_path: Path) -> None:
        spider_list = [Path(spider).stem for spider in glob.glob(f"{str(base_path)}/*.parquet")]
        self.logger.info(f"Found {len(spider_list)} spiders in total")

        spiders_processed_path = self.split_subdir / "spiders_processed.txt"
        if not spiders_processed_path.exists():
            spiders_processed_path.touch()
        spiders_processed = spiders_processed_path.read_text().split("\n")
        self.logger.info(f"Found {len(spiders_processed)} spider(s) already processed: {spiders_processed}")

        spiders_not_yet_processed = set(spider_list) - set(spiders_processed)
        self.logger.info(
            f"Still {len(spiders_not_yet_processed)} spider(s) remaining to process: {spiders_not_yet_processed}")

        for spider in spiders_not_yet_processed:
            self.logger.info(f"Processing spider {spider}")
            df = pd.read_parquet(base_path / (spider + '.parquet'))  # read df of spider
            for lang in self.languages:
                lang_df = df[df.language.str.contains(lang, na=False)]  # select only decisions by language
                if len(lang_df.index) > 0:
                    self.logger.info(f"Processing language {lang}")
                    self.save_chamber_dfs(lang_df)
            with spiders_processed_path.open("a") as f:
                f.write(spider + "\n")

    def save_chamber_dfs(self, lang_df):
        chambers = lang_df.chamber.unique()
        assert len(chambers) > 0  # there should be at least one
        for chamber in chambers:
            chamber_df = lang_df[lang_df.chamber.str.contains(chamber, na=False)]
            path = self.build_path(chamber_df)
            chamber_df.to_parquet(path, index=False)

            gc.collect()  # should prevent memory leak

    def build_path(self, chamber_df):
        language = self.get_level(chamber_df, 'language')
        chamber = self.get_level(chamber_df, 'chamber')
        path = self.split_subdir / language
        path.mkdir(parents=True, exist_ok=True)
        return path / (chamber + ".parquet")

    @staticmethod
    def get_level(chamber_df, level):
        levels = chamber_df[level].unique()
        assert len(levels) == 1
        return levels[0]


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    splitter = Splitter(config)
    splitter.split_spiders()
