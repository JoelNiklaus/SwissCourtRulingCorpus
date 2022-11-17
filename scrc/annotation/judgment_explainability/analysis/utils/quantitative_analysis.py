import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from nltk.tokenize import word_tokenize

import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing

"""
Contains functions for the quantitative analysis. Uses preprocessing.py. Is used by annotation_analysis and occlusion_analysis.
"""
LANGUAGES = ["de", "fr", "it"]
PERSON_NUMBER = {"angela": 1, "lynn": 2, "thomas": 3}
LABELS_OCCLUSION = ["Lower court", "Supports judgment", "Opposes judgment", "Neutral"]
AGGREGATIONS = ["mean", "max", "min"]
GOLD_SESSION = "gold_nina"
LEGAL_AREAS = ["penal_law", "social_law", "civil_law"]
YEARS = ["2015", "2016", "2017", "2018", "2019", "2020"]

EXTRACTED_DATASETS_GOLD = preprocessing.extract_dataset(
    "../legal_expert_annotations/{}/gold/gold_annotations_{}.jsonl",
    "../legal_expert_annotations/{}/gold/gold_annotations_{}-{}.jsonl")
EXTRACTED_DATASETS_MERGED = preprocessing.extract_dataset("../legal_expert_annotations/{}/annotations_{}_merged.jsonl",
                                                          "../legal_expert_annotations/{}/annotations_{}_merged-{}.jsonl")

OCCLUSION_PATHS = {"test_sets": ("../occlusion/lower_court_test_sets/{}/lower_court_test_set_{}.csv",
                                 "../occlusion/occlusion_test_sets/{}/occlusion_test_set_{}_exp_{}.csv"),
                   "prediction": ("../occlusion/lower_court_predictions/{}/predictions_test.csv",
                                  "../occlusion/occlusion_predictions/{}_{}/predictions_test.csv"),
                   "analysis": ("{}/occlusion/bias_lower_court_{}_{}.csv", "{}/occlusion/occlusion_{}_{}_{}.csv")}

NUMBER_OF_EXP = [1, 2, 3, 4]


def remove_baseline(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Returns Dataframe w/o baseline entries.
    """
    return dataset[dataset["explainability_label"] != "Baseline"]


def get_accepted_cases(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Returns accepted cases from dataset.
    """
    return dataset[dataset["answer"] == "accept"]


def count_not_accept(dataset: pd.DataFrame) -> int:
    """
    Returns not accepted cases from dataset.
    """
    return len(dataset[dataset["answer"] != "accept"].groupby("id_csv"))


def count_user_input(dataset: pd.DataFrame) -> int:
    """
    Returns number of user inputs from dataset.
    """
    return len(dataset[dataset["user_input"].notnull()].groupby("id_csv"))


def count_is_correct(dataset: pd.DataFrame) -> (int, int):
    """
    Returns total of number of correct predictions.
    """
    try:
        dataset = get_accepted_cases(dataset)
        return len(dataset[dataset["is_correct"] == True].groupby("id_csv")), len(dataset)
    except KeyError:
        return len(dataset[dataset["is_correct"] == True].groupby("id")), len(dataset)


def get_mean_expl_score(dataset: pd.DataFrame, column: str) -> pd.Series:
    """
    Returns the mean explainability score
    """
    return remove_baseline(dataset).groupby(column).mean()["explainability_score"]


def count_has_flipped(dataset: pd.DataFrame) -> (int, int, int):
    """
    Returns number of flipped experiments, number of unflipped experiments and length of Dataframe.
    """
    dataset = remove_baseline(dataset)
    return len(dataset[dataset["has_flipped"] == True]), len(dataset[dataset["has_flipped"] == False]), len(dataset)


def get_agg_expl_score_occlusion(dataset_0: pd.DataFrame, dataset_1: pd.DataFrame, list_dict: dict) -> dict:
    """
    Applies aggregations to explainability scores.
    Returns a list of dictionaries containing the calculated scores.
    """
    for agg in AGGREGATIONS:
        if f"{agg}_occlusion_0_explainability_score" in list_dict.keys():
            list_dict[f"{agg}_occlusion_0_explainability_score"].append(
                apply_aggregation(dataset_0["explainability_score"], agg))
        else:
            list_dict[f"{agg}_occlusion_0_explainability_score"] = [
                apply_aggregation(dataset_0["explainability_score"], agg)]
        if f"{agg}_occlusion_1_explainability_score" in list_dict.keys():
            list_dict[f"{agg}_occlusion_1_explainability_score"].append(
                apply_aggregation(dataset_1["explainability_score"], agg))
        else:
            list_dict[f"{agg}_occlusion_1_explainability_score"] = [
                apply_aggregation(dataset_1["explainability_score"], agg)]
        return list_dict


def count_approved_dismissed(dataset: pd.DataFrame) -> (int, int):
    """
    Gets all the false dismissal and true approvals.
    Calculates len of approval Dataframe.
    Returns number approved cases and number of dismissed cases.
    """
    false_dismissal = get_accepted_cases(dataset)[get_accepted_cases(dataset)["judgment"] == "dismissal"]
    false_dismissal = false_dismissal[false_dismissal["is_correct"] == False]
    approval = false_dismissal[false_dismissal["judgment"] == "approval"]
    approval = approval[approval["is_correct"] == True]
    approval = len(false_dismissal.append(approval).groupby("id_csv"))
    return approval, abs(len(dataset[dataset["answer"] == "accept"].groupby("id_csv")) - approval)


def apply_aggregation(column: pd.Series, aggregation: str) -> pd.Series:
    """
    Applies aggregation to a specific column
    Returns aggregation result as float.
    """
    if aggregation == "mean":
        return column.mean()
    if aggregation == "max":
        return column.max()
    if aggregation == "min":
        return column.min()


def get_agg_table(annotations: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    @Todo Comment & clean up
    """
    df = pd.DataFrame(index=YEARS, columns=LEGAL_AREAS)
    df.index.name = "year"
    for index, row in annotations.iterrows():
        ser = pd.Series({row["legal_area"]: row[col_name]})
        ser.name = row["year"]
        df = df.append(ser)
    annotations = pd.DataFrame()
    for la in LEGAL_AREAS:
        annotations[la] = df[la].dropna()

    return annotations


def get_total_agg(annotations: pd.DataFrame, aggregation: str, col_name: str):
    """
    @Todo Comment & clean up
    """
    if aggregation == "mean":
        la = annotations.groupby(["legal_area"])[col_name].mean()
        year = annotations.groupby(["year"])[col_name].mean()
    if aggregation == "max":
        la = annotations.groupby(["legal_area"])[col_name].max()
        year = annotations.groupby(["year"])[col_name].max()
    if aggregation == "min":
        la = annotations.groupby(["legal_area"])[col_name].min()
        year = annotations.groupby(["year"])[col_name].min()
    la.name = f"{aggregation}_legal_area"
    return la, year


def count_fact_length(dataset: pd.DataFrame):
    """
    @Todo Comment & clean up
    """
    df_list = []
    annotations = dataset[dataset["answer"] == "accept"]
    annotations["text"] = annotations["text"].str.len()
    annotations["tokens"] = annotations["tokens"].map(len)
    for agg in AGGREGATIONS:
        char = get_agg_table(apply_aggregation(
            annotations.groupby(["year", "legal_area"])["text"], agg).to_frame().reset_index(), "text")
        char = char.append(get_total_agg(annotations, agg, "text")[0])
        char[f"{agg}_year"] = get_total_agg(annotations, agg, "text")[1]
        char.index.name = f"char_{agg}"
        tokens = get_agg_table(apply_aggregation(
            annotations.groupby(["year", "legal_area"])["tokens"], agg).to_frame().reset_index(), "tokens")
        tokens = tokens.append(get_total_agg(annotations, agg, "tokens")[0])
        tokens[f"{agg}_year"] = get_total_agg(annotations, agg, "tokens")[1]
        tokens.index.name = f"tokens_{agg}"
        df_list.append(char)
        df_list.append(tokens)
    return df_list


def count_tokens_per_label(lang: str):
    """
    @Todo Comment & clean up
    """
    pers_mean = {"label": [], "annotator": [], "mean_token": []}
    label_df_list = []
    label_mean = {"label": [], "mean_token": []}
    for label in LABELS_OCCLUSION[:-1]:
        label_df = preprocessing.read_csv(f"{lang}/{label.lower().replace(' ', '_')}_{lang}_3.csv", "index")
        label_df["length"] = 0
        pers_mean[f"{label}_mean_token"] = []
        for person in PERSON_NUMBER.keys():
            try:
                label_df[f"length_{person}"] = label_df[f"tokens_id_{person}"].apply(lambda x: get_length(x))
                label_df["length"].append(label_df[f"length_{person}"])
                pers_mean["label"].append(label)
                pers_mean["annotator"].append(PERSON_NUMBER[person])
                pers_mean["mean_token"].append(label_df[f"length_{person}"].mean())
                pers_mean[f"{label}_mean_token"].append(label_df[f"length_{person}"].mean())
                label_df_list = label_df_list + list(label_df[f"length_{person}"].values)
            except KeyError:
                pass
        label_mean["label"].append(label)
        label_mean["mean_token"].append(float(sum(label_df_list) / len(label_df_list)))
    return pd.DataFrame.from_dict(label_mean), pd.DataFrame.from_dict(
        dict([(k, pd.Series(v)) for k, v in pers_mean.items()]))


def count_lower_court(dataset: pd.DataFrame):
    """
    Returns number of distinct lower courts and their length in tokens.
    """
    dataset["length"] = dataset["lower_court"].apply(lambda x: len(word_tokenize(x)))
    return dataset[dataset["explainability_label"] == "Baseline"]["lower_court"].nunique(), dataset["length"].mean()


def count_occlusion_tokens(lang: str):
    """
    @Todo Comment & clean up
    """
    # Lenght og occluded text --> Version 1 mean length of a sentence
    df_dict = {}
    for nr in NUMBER_OF_EXP:
        occlusion_test_set = preprocessing.read_csv(OCCLUSION_PATHS["test_sets"][1].format(lang, lang, nr), "index")
        occluded_text = occlusion_test_set[occlusion_test_set["explainability_label"] != "None"]
        occluded_text["length"] = occluded_text["occluded_text"].apply(lambda x: len(word_tokenize(x)))
        for agg in AGGREGATIONS:
            if f'{agg}_tokens_occlusion' in df_dict.keys():
                df_dict[f'{agg}_tokens_occlusion'].append(apply_aggregation(occluded_text["length"], agg))
            else:
                df_dict[f'{agg}_tokens_occlusion'] = [apply_aggregation(occluded_text["length"], agg)]

            for label in LABELS_OCCLUSION[1:]:
                if f'{agg}_tokens_occlusion_{label.lower().replace(" ", "_")}' in df_dict.keys():
                    df_dict[f'{agg}_tokens_occlusion_{label.lower().replace(" ", "_")}'].append(
                        apply_aggregation(occluded_text["length"], agg))
                else:
                    df_dict[f'{agg}_tokens_occlusion_{label.lower().replace(" ", "_")}'] = [
                        apply_aggregation(occluded_text["length"], agg)]

    return pd.DataFrame(df_dict)


def count_number_of_sentences(lang: str):
    """
    @Todo Comment & clean up
    """
    occlusion_test_set = preprocessing.read_csv(OCCLUSION_PATHS["test_sets"][1].format(lang, lang, 1), "index")
    return occlusion_test_set.groupby("id").count()["text"].mean()


def get_length(string: str):
    """
    Returns "Nan" for length 0 or length.
    """
    if string != "Nan":
        return len(ast.literal_eval(string))
    else:
        return 0

def write_csv(path: Path, label_mean: pd.DataFrame, pers_mean: pd.DataFrame, df_list: list):
    """
    @Todo Add to preprocessing
    """
    with open(path, "w") as f:
        f.truncate()
        label_mean.to_csv(f)
        f.write("\n")
        pers_mean.to_csv(f)
        f.write("\n")
        for df in df_list:
            df.to_csv(f)
            f.write("\n")


def create_plots(label_mean: pd.DataFrame, pers_mean: pd.DataFrame, df_list: list):
    for df in df_list:
        for agg in AGGREGATIONS:
            try:
                df = df.drop(f"{agg}_legal_area").drop(f"{agg}_year", axis=1)
            except KeyError:
                pass
        # plt.figure()
        # df.plot()
        # plt.show()

    pers_mean = pers_mean.drop(["label", "mean_token"], axis=1).fillna(0)
    pers_mean = pers_mean.groupby("annotator").sum()
    pers_mean.plot(kind='bar')
    plt.show()


def annotation_analysis():
    """
    @Todo Comment & clean up
    """
    for l in LANGUAGES:
        facts_count_list = count_fact_length(EXTRACTED_DATASETS_GOLD[f"annotations_{l}-{GOLD_SESSION}"])
        counts = {"total_number_of_cases":
                      len(get_accepted_cases(EXTRACTED_DATASETS_GOLD[f"annotations_{l}-{GOLD_SESSION}"])),
                  "number_of_comments":
                      count_user_input(EXTRACTED_DATASETS_MERGED[f"annotations_{l}"]),
                  "number_of_not_accepted": count_not_accept(
                      EXTRACTED_DATASETS_GOLD[f"annotations_{l}-{GOLD_SESSION}"]),
                  "number_of_is_correct": count_is_correct(
                      EXTRACTED_DATASETS_GOLD[f"annotations_{l}-{GOLD_SESSION}"])[0],
                  "number_of_approved_prediction":
                      count_approved_dismissed(EXTRACTED_DATASETS_GOLD[f"annotations_{l}-{GOLD_SESSION}"])[0],
                  "number_of_dismissed_prediction":
                      count_approved_dismissed(EXTRACTED_DATASETS_GOLD[f"annotations_{l}-{GOLD_SESSION}"])[1]}
        label_mean, pers_mean = count_tokens_per_label(l)
        preprocessing.write_json(Path(f"{l}/quantitative/annotation_analysis_{l}.json"), counts)
        write_csv(Path(f"{l}/quantitative/annotation_analysis_{l}.csv"), label_mean, pers_mean,
                  facts_count_list)
        create_plots(label_mean, pers_mean, facts_count_list)


def occlusion_analysis():
    """
    @Todo Comment & clean up
    """
    for l in LANGUAGES:
        exp_score_list = []
        lower_court_0, lower_court_1 = preprocessing.read_csv(OCCLUSION_PATHS["analysis"][0].format(l, l, 0), "index"), \
                                       preprocessing.read_csv(OCCLUSION_PATHS["analysis"][0].format(l, l, 1), "index")
        lower_court = lower_court_0.append(lower_court_1)

        mean_lower_court_0, mean_lower_court_1 = get_mean_expl_score(lower_court_0, "lower_court"), get_mean_expl_score(
            lower_court_1, "lower_court")
        exp_score_list.append(mean_lower_court_0.to_frame())
        exp_score_list.append(mean_lower_court_1.to_frame())
        lower_court_test_set = preprocessing.read_csv(OCCLUSION_PATHS["test_sets"][0].format(l, l), "index")

        occlusion_dict = {"number_of_lower_courts": count_lower_court(lower_court_test_set)[0],
                          "mean_tokens_lower_court": count_lower_court(lower_court_test_set)[1],
                          "mean_number_of_sentences_per_case": count_number_of_sentences(l),
                          "mean_lower_court_explainability_score_(0/1)": [
                              apply_aggregation(lower_court_0["explainability_score"], "mean"),
                              apply_aggregation(lower_court_1["explainability_score"], "mean")],
                          "max_lower_court_explainability_score_(0/1)": [
                              apply_aggregation(lower_court_0["explainability_score"], "max"),
                              apply_aggregation(lower_court_1["explainability_score"], "max")],
                          "min_lower_court_explainability_score_(0/1)": [
                              apply_aggregation(lower_court_0["explainability_score"], "min"),
                              apply_aggregation(lower_court_1["explainability_score"], "min")],
                          "number_of_has_flipped_lc(T/F)": list(count_has_flipped(lower_court)),
                          "number_of_is_correct_(lc/total lines))": [
                              count_is_correct(lower_court[lower_court["explainability_label"] != "Baseline"])[0],
                              count_is_correct(lower_court[lower_court["explainability_label"] != "Baseline"])[1]
                          ],
                          "number_of_is_correct_(baseline/total lines))": [
                              count_is_correct(lower_court[lower_court["explainability_label"] == "Baseline"])[0],
                              count_is_correct(lower_court[lower_court["explainability_label"] == "Baseline"])[1]
                          ],
                          }

        exp_score_dict = {}
        for nr in NUMBER_OF_EXP:
            occlusion_0, occlusion_1 = preprocessing.read_csv(OCCLUSION_PATHS["analysis"][1].format(l, nr, l, 0),
                                                              "index"), \
                                       preprocessing.read_csv(OCCLUSION_PATHS["analysis"][1].format(l, nr, l, 1),
                                                              "index")
            occlusion = occlusion_0.append(occlusion_1)
            exp_score_dict = get_agg_expl_score_occlusion(occlusion_0, occlusion_1, exp_score_dict)
            is_correct_list = list(count_is_correct(occlusion[occlusion["explainability_label"] != "Baseline"]))
            occlusion_dict[f"number_of_is_correct_(occlusion_{nr}/total lines)"] = is_correct_list
            mean_occlusion_0, mean_occlusion_1 = get_mean_expl_score(occlusion_0,
                                                                     "explainability_label"), get_mean_expl_score(
                occlusion_1, "explainability_label")
            exp_score_list.append(mean_occlusion_0.to_frame())
            exp_score_list.append(mean_occlusion_1.to_frame())

        preprocessing.write_json(Path(f"{l}/quantitative/occlusion_analysis_{l}.json"), occlusion_dict)
        write_csv(Path(f"{l}/quantitative/occlusion_analysis_{l}.csv"), count_occlusion_tokens(l),
                  pd.DataFrame(exp_score_dict), exp_score_list)
