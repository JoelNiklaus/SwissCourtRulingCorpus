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

        self.debug_chunksize = 1000

    def split_BGE_text(self, example):
        regeste_keyword = "\nRegeste"
        facts_keyword = "\nSachverhalt"
        considerations_keyword = "\nErwägungen"

        # split into header and rest
        regeste_split = example["full_text"].split(regeste_keyword, 1)
        example["header"] = regeste_split[0]

        if len(regeste_split) < 2:
            self.logger.warning(f"Could not split text for {example['file_name']}")
            example["regeste"] = None
            example["text"] = None
            return example

        # split into regeste and text
        facts_split = regeste_split[1].split(facts_keyword, 1)
        regeste_based_on_facts_split = regeste_keyword + facts_split[0]
        text_based_on_facts_split = (facts_keyword + facts_split[1]) if len(facts_split) > 1 else None

        considerations_split = regeste_split[1].split(considerations_keyword, 1)
        regeste_based_on_considerations_split = regeste_keyword + considerations_split[0]
        text_based_on_considerations_split = (considerations_keyword + considerations_split[1]) if len(
            considerations_split) > 1 else None

        if text_based_on_facts_split is None and text_based_on_considerations_split is None:
            self.logger.warning(f"Could not split text for {example['file_name']}")
            example["regeste"] = None
            example["text"] = None
        elif text_based_on_facts_split is None:  # if the text is empty, use the other text
            example["text"] = text_based_on_considerations_split
            example["regeste"] = regeste_based_on_considerations_split
        elif text_based_on_considerations_split is None:  # if the text is empty, use the other text
            example["text"] = text_based_on_facts_split
            example["regeste"] = regeste_based_on_facts_split
        else:
            # take the one where the regeste is shorter
            # because it is more likely that a wrong keyword is found in the much longer text than the regeste
            if len(regeste_based_on_facts_split) < len(regeste_based_on_considerations_split):
                example["text"] = text_based_on_facts_split
                example["regeste"] = regeste_based_on_facts_split
            else:
                example["text"] = text_based_on_considerations_split
                example["regeste"] = regeste_based_on_considerations_split

        return example

    def prepare_dataset(self, save_reports, court_string):
        engine = self.get_engine(self.db_scrc)
        data_to_load = {
            "section": True, "file": True, "file_number": False,
            "judgment": False, "citation": True, "lower_court": False,
            "law_area": True, "law_sub_area": True
        }
        # we don't use the cache since it is overwritten after each court
        df = self.get_df(engine, data_to_load, court_string=court_string, use_cache=False)
        hf_dataset = datasets.Dataset.from_pandas(df)

        hf_dataset = hf_dataset.map(self.split_BGE_text, num_proc=4)
        self.logger.info(f"Number of examples after split: {len(hf_dataset)}")
        # filter out examples where either the regeste or the text is None
        hf_dataset = hf_dataset.filter(lambda example: example is not None
                                                       and example["regeste"] is not None
                                                       and example["text"] is not None)
        self.logger.info(f"Number of examples after filtering: {len(hf_dataset)}")

        # TODO try to get HTML and see if there more information is available allowing for cleaner splits

        return hf_dataset, None


if __name__ == '__main__':
    config = get_config()

    pretraining_dataset_creator = RegesteDatasetCreator(config, debug=False)
    court_list = ["CH_BGE"]
    pretraining_dataset_creator.create_multiple_datasets(court_list=court_list, concatenate=False, overview=True,
                                                         sub_datasets=False, save_reports=False)
