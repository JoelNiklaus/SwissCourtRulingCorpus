import datasets

from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.enums.section import Section
from scrc.utils.log_utils import get_logger

from scrc.utils.main_utils import get_config
import scrc.utils.monkey_patch  # IMPORTANT: DO NOT REMOVE: prevents memory leak with pandas


class RegesteDatasetCreator(DatasetCreator):
    """
    Creates a dataset with all the full_text
    """

    def __init__(self, config: dict, debug: bool = True):
        super().__init__(config, debug)
        self.logger = get_logger(__name__)

        self.overwrite_cache = True
        self.split_type = "date-stratified"
        self.dataset_name = "regeste"
        self.feature_cols = [Section.FULL_TEXT]

    def prepare_dataset(self, save_reports, court_string):
        engine = self.get_engine(self.db_scrc)
        data_to_load = {
            "section": True, "file": False, "file_number": False,
            "judgment": False, "citation": True, "lower_court": False,
            "law_area": True, "law_sub_area": True
        }
        # we don't use the cache since it is overwritten after each court
        df = self.get_df(engine, data_to_load, court_string=court_string)
        hf_dataset = datasets.Dataset.from_pandas(df)
        print(df.head(1))

        # TODO try to get HTML and see if there more information is available
        # TODO: 1. Search for Regeste/Regesto 2. Take everything until Sachverhalt/Faits? etc. as summary 3. Take everything after as text

        return hf_dataset, None


if __name__ == '__main__':
    config = get_config()

    pretraining_dataset_creator = RegesteDatasetCreator(config, debug=True)
    court_list = ["CH_BGE"]
    pretraining_dataset_creator.create_multiple_datasets(court_list=court_list, concatenate=False, overview=True,
                                                         sub_datasets=False, save_reports=False)
