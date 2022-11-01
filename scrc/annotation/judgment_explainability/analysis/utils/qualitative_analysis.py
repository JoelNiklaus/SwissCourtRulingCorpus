from pathlib import Path

import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing

LANGUAGES = ["de", "fr", "it"]
SESSIONS = ["angela", "lynn", "thomas", "gold_final", "gold_nina"]
GOLD_SESSION = {"de": "gold_final", "fr": "gold_nina", "it": "gold_nina"}
ANNOTATION_PATHS = [
    ("../legal_expert_annotations/{}/annotations_{}.jsonl", "../legal_expert_annotations/{}/annotations_{}-{}.jsonl"),
    ("../legal_expert_annotations/{}/annotations_{}_inspect.jsonl",
     "../legal_expert_annotations/{}/annotations_{}_inspect-{}.jsonl"),
    ("../legal_expert_annotations/{}/annotations_{}_merged.jsonl",
     "../legal_expert_annotations/{}/annotations_{}_merged-{}.jsonl"),
    ("../legal_expert_annotations/{}/gold/gold_annotations_{}.jsonl",
     "../legal_expert_annotations/{}/gold/gold_annotations_{}-{}.jsonl")]


def dump_user_input(datasets: dict, lang: str, version: str, ):
    """
    Dumps all the user inputs and not accepted cases as csv.
    Catches key errors.
    """
    for session in SESSIONS:
        try:
            user_input = datasets[f"annotations_{lang}-{session}"][
                datasets[f"annotations_{lang}-{session}"]["user_input"].notnull()]
            preprocessing.write_csv(Path(f"{lang}/qualitative/user_input_{lang}-{session}_{version}.csv"),
                                    user_input[['id_scrc', '_annotator_id', "user_input"]])
            print(f"Saved user_input_{lang}-{session}_{version}.csv successfully!")
            case_not_accepted = datasets[f"annotations_{lang}-{session}"][
                datasets[f"annotations_{lang}-{session}"]["answer"] != "accept"]
            preprocessing.write_csv(Path(f"{lang}/qualitative/ig_re_{lang}-{session}_{version}.csv"),
                                    case_not_accepted[['id_scrc', '_annotator_id', "user_input"]])
            print(f"Saved ig_re_{lang}-{session}_{version}.csv successfully!")
        except KeyError:
            pass


def analysis():
    for paths in ANNOTATION_PATHS:
        extracted_datasets = preprocessing.extract_dataset(paths[0], paths[1])
        for l in LANGUAGES:
            for ver in ["1", "2", "3", "4"]:
                dump_user_input(extracted_datasets, l, ver)
