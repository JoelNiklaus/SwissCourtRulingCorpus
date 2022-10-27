import ast
import os
from pathlib import Path

from scrc.annotation.judgment_explainability.analysis.preprocessing import extract_dataset, write_csv

LANGUAGES = ast.literal_eval(os.getenv("LANGUAGES"))
SESSIONS = ast.literal_eval(os.getenv("SESSIONS"))
GOLD_SESSION = {"de":"gold_final", "fr":"gold_nina", "it":"gold_nina"}


def dump_user_input(datasets: dict, lang: str, version: str, ):
    """
    Dumps all the user inputs and not accepted cases as csv.
    Catches key errors.
    """
    for session in SESSIONS:
        try:
            user_input = datasets[f"annotations_{lang}-{session}"][
                datasets[f"annotations_{lang}-{session}"]["user_input"].notnull()]
            write_csv(Path(f"{lang}/qualitative/user_input_{lang}-{session}_{version}.csv"),
                      user_input[['id_scrc', '_annotator_id', "user_input"]])
            print(f"Saved user_input_{lang}-{session}_{version}.csv successfully!")
            case_not_accepted = datasets[f"annotations_{lang}-{session}"][
                datasets[f"annotations_{lang}-{session}"]["answer"] != "accept"]
            write_csv(Path(f"{lang}/qualitative/ig_re_{lang}-{session}_{version}.csv"),
                      case_not_accepted[['id_scrc', '_annotator_id', "user_input"]])
            print(f"Saved ig_re_{lang}-{session}_{version}.csv successfully!")
        except KeyError:
            pass


def analysis():
    extracted_datasets_gold =  extract_dataset("../legal_expert_annotations/{}/gold/gold_annotations_{}.jsonl","../legal_expert_annotations/{}/gold/gold_annotations_{}-{}.jsonl")
    extracted_datasets_1 = extract_dataset("../legal_expert_annotations/{}/annotations_{}.jsonl", "../legal_expert_annotations/{}/annotations_{}-{}.jsonl")
    extracted_datasets_2 = extract_dataset("../legal_expert_annotations/{}/annotations_{}_inspect.jsonl", "../legal_expert_annotations/{}/annotations_{}_inspect-{}.jsonl")
    extracted_datasets_3 = extract_dataset("../legal_expert_annotations/{}/annotations_{}_merged.jsonl",
                                           "../legal_expert_annotations/{}/annotations_{}_merged-{}.jsonl")


    for l in LANGUAGES:
        dump_user_input(extracted_datasets_1,l, "1")
        dump_user_input(extracted_datasets_2,l, "2")
        dump_user_input(extracted_datasets_3,l, "3")
        dump_user_input(extracted_datasets_gold, l, "4")
