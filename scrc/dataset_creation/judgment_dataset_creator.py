import configparser

from root import ROOT_DIR
from scrc.dataset_creation.dataset_creator import DatasetCreator
from scrc.utils.log_utils import get_logger
import numpy as np


class JudgmentDatasetCreator(DatasetCreator):
    """
    Creates a dataset with the facts or considerations as input and the judgements as labels
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)

        self.debug = False
        self.split_type = "date-stratified"
        self.dataset_name = "judgment_prediction"
        self.feature_cols = ['facts', 'considerations']

        self.with_write_off = False
        self.with_unification = False
        self.with_inadmissible = False
        self.with_partials = False
        self.make_single_label = True

    def get_dataset(self, feature_col, lang, save_reports):
        df = self.get_df(self.get_engine(self.db_scrc), feature_col, 'judgements', lang, save_reports)

        # Delete cases with "Nach Einsicht" from the dataset because they are mostly inadmissible or otherwise dismissal
        # => too easily learnable for the model (because of spurious correlation)
        if self.with_inadmissible:
            df = df[~df[feature_col].str.startswith('Nach Einsicht')]

        def clean(judgements):
            out = set()
            for judgement in judgements:
                # remove "partial_" from all the items to merge them with full ones
                if not self.with_partials:
                    judgement = judgement.replace("partial_", "")

                out.add(judgement)

            if not self.with_write_off:
                # remove write_off because reason for it happens mostly behind the scenes and not written in the facts
                if 'write_off' in judgements:
                    out.remove('write_off')

            if not self.with_unification:
                # remove unification because reason for it happens mostly behind the scenes and not written in the facts
                if 'unification' in judgements:
                    out.remove('unification')

            if not self.with_inadmissible:
                # remove inadmissible because it is a formal reason and not that interesting semantically.
                # Facts are formulated/summarized in a way to justify the decision of inadmissibility
                # hard to know solely because of the facts (formal reasons, not visible from the facts)
                if 'inadmissible' in judgements:
                    out.remove('inadmissible')

            # remove all labels which are complex combinations (reason: different questions => model cannot know which one to pick)
            if self.make_single_label:
                # contrary judgements point to multiple questions which is too complicated
                if 'dismissal' in out and 'approval' in out:
                    return np.nan
                # if we have inadmissible and another one, we just remove inadmissible
                if 'inadmissible' in out and len(out) > 1:
                    out.remove('inadmissible')
                if len(out) > 1:
                    message = f"By now we should only have one label. But instead we still have the labels {out}"
                    raise ValueError(message)
                elif len(out) == 1:
                    return out.pop()  # just return the first label because we only have one left
                else:
                    return np.nan

            return list(out)

        df = df.dropna(subset=['judgements'])
        df.judgements = df.judgements.apply(clean)
        df = df.dropna(subset=['judgements'])  # drop empty labels introduced by cleaning before

        df = df.rename(columns={feature_col: "text", "judgements": "label"})  # normalize column names
        labels, _ = list(np.unique(np.hstack(df.label), return_index=True))
        return df, labels


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(ROOT_DIR / 'config.ini')  # this stops working when the script is called from the src directory!

    judgment_dataset_creator = JudgmentDatasetCreator(config)
    judgment_dataset_creator.create_dataset(sub_datasets=False, kaggle=False, huggingface=True, save_reports=False)
