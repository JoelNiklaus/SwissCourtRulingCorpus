from pathlib import Path

import pandas as pd

import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing

"""
@Todo maybe read some file already up here
"""
LANGUAGES = ["de", "fr", "it"]
SESSIONS = ["angela", "lynn", "thomas", "gold_nina"]
ANNOTATION_PATHS = {
    "1": ("../legal_expert_annotations/{}/annotations_{}.jsonl",
          "../legal_expert_annotations/{}/annotations_{}-{}.jsonl"),
    "2": ("../legal_expert_annotations/{}/annotations_{}_inspect.jsonl",
          "../legal_expert_annotations/{}/annotations_{}_inspect-{}.jsonl"),
    "3": ("../legal_expert_annotations/{}/annotations_{}_merged.jsonl",
          "../legal_expert_annotations/{}/annotations_{}_merged-{}.jsonl"),
    "4": ("../legal_expert_annotations/{}/gold/gold_annotations_{}.jsonl",
          "../legal_expert_annotations/{}/gold/gold_annotations_{}-{}.jsonl")}

OCCLUSION_PATHS = {"test_sets": ("../occlusion/lower_court_test_sets/{}/lower_court_test_set_{}.csv",
                                 "../occlusion/occlusion_test_sets/{}/occlusion_test_set_{}_exp_{}.csv"),
                   "prediction": ("../occlusion/lower_court_predictions/{}/predictions_test.csv",
                                  "../occlusion/occlusion_predictions/{}_{}/predictions_test.csv"),
                   "analysis": ("{}/occlusion/bias_lower_court_{}_{}.csv", "{}/occlusion/occlusion_{}_{}_{}.csv")}
NUMBER_OF_EXP = [1, 2, 3, 4]


def dump_user_input(datasets: dict, lang: str, version: str, ):
    """
    Dumps all the user inputs and not accepted cases as csv.
    Catches key errors.
    """
    for session in SESSIONS[:-1]:
        try:
            user_input = datasets[f"annotations_{lang}-{session}"][
                datasets[f"annotations_{lang}-{session}"]["user_input"].notnull()]
            preprocessing.write_csv(f"{lang}/qualitative/user_input_{lang}-{session}_{version}.csv",
                                    user_input[['id_scrc', '_annotator_id', "user_input"]])
            case_not_accepted = datasets[f"annotations_{lang}-{session}"][
                datasets[f"annotations_{lang}-{session}"]["answer"] != "accept"]
            preprocessing.write_csv(f"{lang}/qualitative/ig_re_{lang}-{session}_{version}.csv",
                                    case_not_accepted[['id_scrc', '_annotator_id', "user_input"]])
        except KeyError:
            pass


def get_flipped_column(dataset: pd.DataFrame, column: str):
    """
    Returns dataset of only flipped cases and counts per column of flipped cases.
    """
    dataset = dataset[dataset["explainability_label"] != "Baseline"]
    dataset = dataset[dataset["has_flipped"] == True]
    counts = dataset[column].value_counts()
    return dataset, counts


def annotation_analysis():
    for l in LANGUAGES:
        for ver in ["3", "4"]:
            extracted_datasets = preprocessing.extract_dataset(ANNOTATION_PATHS[ver][0], ANNOTATION_PATHS[ver][1])
            dump_user_input(extracted_datasets, l, ver)


def occlusion_analysis():
    """
    @Todo Clean up, comment
    """
    json_dict = {}
    for l in LANGUAGES:
        lower_court_0, lower_court_1 = preprocessing.read_csv(OCCLUSION_PATHS["analysis"][0].format(l, l, 0), "index"), \
                                       preprocessing.read_csv(OCCLUSION_PATHS["analysis"][0].format(l, l, 1), "index")
        lower_court = lower_court_0.append(lower_court_1)
        json_dict[f"lower_court"] = get_flipped_column(lower_court, "lower_court")[1].to_dict()
        json_dict[f"number_of_flipped_experiments_lower_court"] = len(get_flipped_column(lower_court, "lower_court")[0])
        json_dict[f"number_of_flipped_cases_lower_court"] = len(
            get_flipped_column(lower_court, "lower_court")[0]["id"].unique())
        df_list = [get_flipped_column(lower_court_0, "lower_court")[0],
                   get_flipped_column(lower_court_1, "lower_court")[0]]

        for nr in NUMBER_OF_EXP:
            occlusion_0, occlusion_1 = preprocessing.read_csv(OCCLUSION_PATHS["analysis"][1].format(l, nr, l, 0),
                                                              "index"), \
                                       preprocessing.read_csv(OCCLUSION_PATHS["analysis"][1].format(l, nr, l, 1),
                                                              "index")
            occlusion = occlusion_0.append(occlusion_1)

            df_list.append(get_flipped_column(occlusion_0, "explainability_label")[0])
            df_list.append(get_flipped_column(occlusion_1, "explainability_label")[0])
            if "occlusion" in json_dict.keys():
                json_dict["occlusion"].append(get_flipped_column(occlusion, "explainability_label")[1].to_dict())
                json_dict["number_of_flipped_experiments_occlusion"][
                    f"number_of_flipped_experiments_occlusion_{nr}"] = len(
                    get_flipped_column(occlusion, "explainability_label")[0])
                json_dict[f"number_of_flipped_cases_occlusion"][f"number_of_flipped_cases_occlusion_{nr}"] = len(
                    get_flipped_column(occlusion, "explainability_label")[0]["id"].unique())
            else:
                json_dict["occlusion"] = [get_flipped_column(occlusion, "explainability_label")[1].to_dict()]
                json_dict["number_of_flipped_experiments_occlusion"] = {
                    f"number_of_flipped_experiments_occlusion_{nr}": len(
                        get_flipped_column(occlusion, "explainability_label")[0])}
                json_dict[f"number_of_flipped_cases_occlusion"] = {f"number_of_flipped_cases_occlusion_{nr}": len(
                    get_flipped_column(occlusion, "explainability_label")[0]["id"].unique())}

        preprocessing.write_csv_from_list(f"{l}/qualitative/occlusion_analysis.csv", df_list)
        preprocessing.write_json(f"{l}/qualitative/occlusion_analysis.json", json_dict)
