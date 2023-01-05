import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from scipy.stats import sem

import scrc.annotation.judgment_explainability.analysis.utils.plots as plots
import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing
from scrc.annotation.judgment_explainability.analysis.utils import scores

COLORS = {"red": "#883955".lower(), "yellow": "#EFEA5A".lower(), "green": "#83E377".lower(), "blue": "#2C699A".lower(),
          "purple": "#54478C".lower(), "grey": "#737370".lower(),
          "turquoise": "#0DB39E".lower(), "dark blue": "#163650".lower(), "dark green": "#25691c".lower(),
          "light purple": "#8477bb"}

warnings.filterwarnings("ignore")

"""
Contains functions for the quantitative analysis. Uses preprocessing.py and plots.py. 
Is used by annotation_analysis, lower_court and occlusion_analysis.
"""
LANGUAGES = ["de", "fr", "it"]
PERSONS = ["angela", "lynn", "thomas"]
PERSON_NUMBER = {"angela": 1, "lynn": 2, "thomas": 3}
LABELS_OCCLUSION = ["Lower court", "Supports judgment", "Opposes judgment", "Neutral"]
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
AGGREGATIONS = ["mean", "max", "min"]
GOLD_SESSION = "gold_nina"
LEGAL_AREAS = ["penal_law", "social_law", "civil_law"]
YEARS = ["2015", "2016", "2017", "2018", "2019", "2020"]
NUMBER_OF_EXP = [1, 2, 3, 4]
IAA_List = ["1_2", "1_3", "2_3"]
SCORES_COLUMNS = ['overlap_maximum', 'overlap_minimum', 'jaccard_similarity', 'meteor_score', 'bleu_score', 'rouge1',
                  'rouge2', 'rougeL', 'F1']
LOWER_COURT_ABBREVIATIONS = preprocessing.read_json("lower_court_abbrevation.json")
preprocessing.write_csv("lower_court_abbrevation.csv", pd.DataFrame.from_dict(LOWER_COURT_ABBREVIATIONS))

EXTRACTED_DATASETS_GOLD = preprocessing.extract_dataset(
    "../legal_expert_annotations/{}/gold/gold_annotations_{}.jsonl",
    "../legal_expert_annotations/{}/gold/gold_annotations_{}-{}.jsonl")
EXTRACTED_DATASETS_MERGED = preprocessing.extract_dataset("../legal_expert_annotations/{}/annotations_{}_merged.jsonl",
                                                          "../legal_expert_annotations/{}/annotations_{}_merged-{}.jsonl")
OCCLUSION_PATHS = {"test_sets": ("../occlusion/lower_court_test_sets/{}/lower_court_test_set_{}.csv",
                                 "../occlusion/occlusion_test_sets/{}/occlusion_test_set_{}_exp_{}.csv"),
                   "prediction": ("../occlusion/lower_court_predictions/{}/predictions_test.csv",
                                  "../occlusion/occlusion_predictions/{}_{}/predictions_test.csv"),
                   "analysis": ("{}/occlusion/bias_lower_court_{}.csv", "{}/occlusion/occlusion_{}_{}.csv")}


def count_user_input(dataset: pd.DataFrame) -> int:
    """
    Returns number of user inputs from dataset.
    """
    return len(dataset[dataset["user_input"].notnull()].groupby("id_csv"))


def count_not_accept(dataset: pd.DataFrame) -> int:
    """
    Returns not accepted cases from dataset.
    """
    return len(dataset[dataset["answer"] != "accept"].groupby("id_csv"))


def count_is_correct(dataset: pd.DataFrame) -> (int, int):
    """
    Returns total of number of correct predictions.
    """
    return len(dataset[dataset["is_correct"] == True].groupby("id_csv")), len(dataset)


def count_approved_dismissed(dataset: pd.DataFrame) -> (int, int):
    """
    Returns number approved cases and number of dismissed cases.
    """
    try:
        return len(dataset[dataset["judgment"] == "approval"].groupby("id_csv")), \
           len(dataset[dataset["judgment"] == "dismissal"].groupby("id_csv"))
    except KeyError:
        return len(dataset[dataset["prediction"] == 1]), \
               len(dataset[dataset["prediction"] == 0])



def apply_len_occlusion_chunks(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Returns length of occluded text.
    """
    dataset["length"] = dataset["occluded_text"].apply(lambda x: len(word_tokenize(x)))
    return dataset


def get_number_of_sentences(dataset: pd.DataFrame) -> int:
    """
    Returns mean number of sentences.
    """
    return dataset.groupby("id").count().reset_index()["prediction"].mean()


def count_lower_court(dataset: pd.DataFrame) -> (int, float):
    """
    Returns number of distinct lower courts and their length in tokens.
    """
    dataset["length"] = dataset["lower_court_long"].apply(lambda x: len(word_tokenize(x)))
    return dataset["lower_court_long"].nunique(), dataset["length"].mean()


def count_has_flipped(dataset: pd.DataFrame) -> (int, int, int):
    """
    Returns number of flipped experiments, number of unflipped experiments and length of Dataframe.
    """
    return len(dataset[dataset["has_flipped"] == True]), len(dataset[dataset["has_flipped"] == False]), len(dataset)


def abbreviate_lower_courts(lang: str, row: str) -> str:
    """
    Returns abbreviation for each lower court.
    """
    for key in LOWER_COURT_ABBREVIATIONS[lang].keys():
        if row == key:
            return f"{LOWER_COURT_ABBREVIATIONS[lang][key]}"


def get_annotation_label_mean(lang: str) -> pd.DataFrame:
    """
    Gets number of tokens of each explainability label.
    Calculates mean token number and error of each explainability label.
    Returns Dataframe.
    """
    label_df_list = []
    label_mean = {"label": [], "mean_token": [], "error": []}
    for label in LABELS_OCCLUSION:
        label_df = preprocessing.read_csv(f"{lang}/{label.lower().replace(' ', '_')}_{lang}_4.csv", "index")
        label_df["length"] = label_df["tokens_id"].apply(lambda x: get_length(x))
        label_df_list = label_df_list + list(label_df[f"length"].values)
        label_mean["error"].append(sem(label_df_list))
        label_mean["label"].append(label)
        label_mean["mean_token"].append(
            float(sum(label_df_list) / len(list(filter(lambda num: num != 0, label_df_list)))))
    return pd.DataFrame.from_dict(label_mean)


def get_occlusion_label_mean(dataset) -> (float, pd.DataFrame):
    """
    Gets number of tokens of each explainability label.
    Calculates mean token number and error of each explainability label.
    Returns float and Dataframe.
    """
    mean_length_per_label_dict = {}
    mean_chunk_length = []
    for label in LABELS_OCCLUSION[1:]:
        label_series = apply_len_occlusion_chunks(dataset[dataset["explainability_label"] == label])["length"]
        error = sem(list(label_series.values))
        mean_chunk_length = mean_chunk_length + label_series.to_list()
        mean_length_per_label_dict[label] = [label_series.mean()]
        mean_length_per_label_dict[f"{label}_error"] = error

    return float(np.mean(mean_chunk_length)), pd.DataFrame.from_dict(mean_length_per_label_dict)


def get_annotation_pers_mean(lang: str) -> pd.DataFrame:
    """
    Gets number of tokens of each explainability label per person and case.
    Calculates mean token number of each explainability label per person.
    Returns Dataframe.
    """
    pers_mean = {"label": [], "annotator": [], "mean_token": [], }
    for label in LABELS:
        label_df = preprocessing.read_csv(f"{lang}/{label.lower().replace(' ', '_')}_{lang}_3.csv", "index")
        pers_mean[f"{label.lower().replace(' ', '_')}_mean_token"] = []
        pers_mean[f"{label.lower().replace(' ', '_')}_error"] = []
        label_mean = []
        for person in PERSON_NUMBER.keys():
            try:
                label_df[f"length_{PERSON_NUMBER[person]}"] = label_df[f"tokens_id_{person}"].apply(
                    lambda x: get_length(x))
                pers_mean["label"].append(label)
                pers_mean["annotator"].append(PERSON_NUMBER[person])
                pers_mean[f"{label.lower().replace(' ', '_')}_error"].append(
                    sem(list(label_df[f"length_{PERSON_NUMBER[person]}"].dropna().values)))
                pers_mean["mean_token"].append(label_df[f"length_{PERSON_NUMBER[person]}"].mean())
                pers_mean[f"{label.lower().replace(' ', '_')}_mean_token"].append(
                    label_df[f"length_{PERSON_NUMBER[person]}"].mean())
                label_mean = label_mean + list(label_df[f"length_{PERSON_NUMBER[person]}"].dropna().values)
            except KeyError:
                pass
        pers_mean[f"{label.lower().replace(' ', '_')}_mean"] = float(
            sum(label_mean) / len(list(filter(lambda num: num != 0, label_mean))))

    return pd.DataFrame.from_dict(dict([(k, pd.Series(v)) for k, v in pers_mean.items()]))


def get_length(string: str):
    """
    Returns "Nan" for length 0 or length.
    """
    if string != "Nan":
        return len(ast.literal_eval(string))
    else:
        return np.NaN


def count_mean_fact_length(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Gets facts per case, calculates mean token length per case.
    Returns Dataframe.
    """
    dataset.loc[:, "tokens"] = dataset["tokens"].map(len)
    tokens_df = reshape_df(dataset.groupby(["year", "legal_area"])["tokens"].mean().to_frame().reset_index(),
                           "tokens")
    tokens_df = tokens_df.append(get_mean_col(dataset, "tokens")[0])
    tokens_df[f"mean_year"] = get_mean_col(dataset, "tokens")[1]
    tokens_df.index.name = f"tokens_mean"
    tokens_df.at['mean_legal_area', 'mean_year'] = (tokens_df['mean_year'].mean() + tokens_df.loc[
        'mean_legal_area'].mean()) / 2
    return tokens_df


def reshape_df(df_2: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Reshapes Dataframe from:
    i c_1 c_2 c_3       c_1 0_2 1_2
    0  y  0_2 0_3  to:  y  0_3 1_3
    1  y  1_2 1_3
    Returns reshaped Dataframe
    """
    df_1 = pd.DataFrame(index=YEARS, columns=LEGAL_AREAS)
    df_1.index.name = "year"
    for index, row in df_2.iterrows():
        ser = pd.Series({row["legal_area"]: row[col_name]})
        ser.name = row["year"]
        df_1 = df_1.append(ser)
    df_2 = pd.DataFrame()
    for la in LEGAL_AREAS:
        df_2[la] = df_1[la].dropna()
    return df_2


def get_mean_col(dataset: pd.DataFrame, col_name: str) -> (pd.Series, pd.Series):
    """
    Groups column by legal_area and year, applies mean.
    Returns Series.
    """
    la = dataset.groupby(["legal_area"])[col_name].mean()
    year = dataset.groupby(["year"])[col_name].mean()
    la.name = f"mean_legal_area"
    return la, year


def get_legal_area_distribution(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates proportional distribution of cases per legal area.
    Returns Dataframe.
    """
    distribution_df = dataset.groupby("legal_area").count().reset_index()[["legal_area", "id_scrc"]]
    distribution_df["distribution"] = distribution_df["id_scrc"].div(len(dataset))
    return distribution_df.drop("id_scrc", axis=1, inplace=False)


def get_lc_distribution(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Counts proportional occurrences of each lower court.
    Returns Dataframe
    """
    distribution_df = dataset.groupby("lower_court")["id"].count().reset_index() \
        .rename(columns={"index": "lower_court", "id": "count"})
    distribution_df["count"] = distribution_df["count"].div(distribution_df["count"].sum())
    return distribution_df


def get_lc_la_distribution(dataset: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Groups by lower_court and applies list() to rows.
    Gets mean confidence_scaled per legal_area and distribution over legal areas.
    Returns Dataframes.
    """
    legal_area_df = preprocessing.group_by_agg_column(dataset, "lower_court",
                                                      agg_dict={"id": "count", "confidence_scaled": lambda x: list(x),
                                                                "legal_area": lambda x: list(x)})
    conf_df = get_mean_count_la(legal_area_df, "conf_legal_area", get_conf_dict_legal_area)
    for la in LEGAL_AREAS:
        conf_df[la] = conf_df[la].apply(np.mean)
    distribution_df = get_mean_count_la(legal_area_df, "count_legal_area", get_count_dict_legal_area)
    distribution_df[LEGAL_AREAS] = distribution_df[LEGAL_AREAS].div(legal_area_df["id"], axis=0)
    return distribution_df[["lower_court"] + LEGAL_AREAS], conf_df[["lower_court"] + LEGAL_AREAS]


def get_mean_count_la(dataset: pd.DataFrame, col: str, func):
    """
    Applies either get_conf_dict_legal_area or get_count_dict_legal_area to rows.
    Returns Dataframe.
    """
    dataset[col] = dataset.apply(lambda row: func(
        {row["confidence_scaled"][i]: row["legal_area"][i] for i in range(len(row["confidence_scaled"]))}, ), axis=1)
    dataset[LEGAL_AREAS] = dataset[col].apply(pd.Series)
    return dataset


def get_conf_dict_legal_area(conf_legal_area_dict: dict) -> dict:
    """
    Returns dict of shape {"legal_area_1": [conf_1, conf_2, ...]}
    """
    legal_area_dict = {i: [] for i in LEGAL_AREAS}
    for k in conf_legal_area_dict.keys():
        if conf_legal_area_dict[k] == LEGAL_AREAS[0]:
            legal_area_dict[LEGAL_AREAS[0]].append(float(k))
        if conf_legal_area_dict[k] == LEGAL_AREAS[1]:
            legal_area_dict[LEGAL_AREAS[1]].append(float(k))
        if conf_legal_area_dict[k] == LEGAL_AREAS[2]:
            legal_area_dict[LEGAL_AREAS[2]].append(float(k))
    return legal_area_dict


def get_count_dict_legal_area(conf_legal_area_dict):
    """
    Returns dict of shape {legal_area_1: count_legal_area}
    """
    legal_area_dict = {i: 0 for i in LEGAL_AREAS}
    for k in conf_legal_area_dict.keys():
        if conf_legal_area_dict[k] == LEGAL_AREAS[0]:
            legal_area_dict[LEGAL_AREAS[0]] += 1
        if conf_legal_area_dict[k] == LEGAL_AREAS[1]:
            legal_area_dict[LEGAL_AREAS[1]] += 1
        if conf_legal_area_dict[k] == LEGAL_AREAS[2]:
            legal_area_dict[LEGAL_AREAS[2]] += 1
    return legal_area_dict


def split_oppose_support_false(dataset_1: pd.DataFrame, dataset_2: pd.DataFrame):
    """
    @ Todo
    """
    false_classification_pos, false_classification_neg = dataset_2[
                                                             dataset_2["confidence_direction"] == 1], \
                                                         dataset_2[
                                                             dataset_2["confidence_direction"] == -1]
    o_judgement, s_judgement = dataset_1[dataset_1["numeric_label"] == 1], dataset_1[
        dataset_1["numeric_label"] == -1]
    o_judgement, s_judgement = o_judgement[["id", "explainability_label", "numeric_label", "occluded_text"]].drop_duplicates(), \
                               s_judgement[["id", "explainability_label", "numeric_label", "occluded_text"]].drop_duplicates()

    o_judgement, s_judgement = pd.merge(false_classification_pos, o_judgement, on="id",
                                        suffixes=(f'_model', f'_human'),
                                        how="inner").drop_duplicates(), \
                               pd.merge(false_classification_neg, s_judgement, on="id",
                                        suffixes=(f'_model', f'_human'),
                                        how="inner").drop_duplicates()
    return o_judgement, s_judgement


def split_oppose_support_correct(dataset_2: pd.DataFrame):
    """
    @ Todo
    """
    correct_classification_pos, correct_classification_neg = dataset_2[
                                                             dataset_2["confidence_direction"] == 1], \
                                                         dataset_2[
                                                             dataset_2["confidence_direction"] == -1]
    rename_dict = {col: f"{col}_human" for col in ['explainability_label', 'occluded_text', 'numeric_label']}
    rename_list = [f"{col}_model" for col in ['explainability_label', 'occluded_text', 'numeric_label']]
    correct_classification_pos[rename_list] = correct_classification_pos[['explainability_label', 'occluded_text', 'numeric_label']]
    correct_classification_neg[rename_list] = correct_classification_neg[['explainability_label', 'occluded_text', 'numeric_label']]
    return correct_classification_pos.rename(rename_dict, axis=1), correct_classification_neg.rename(rename_dict, axis=1)


def apply_occ_ttest(baseline_df: pd.DataFrame, occlusion_df: pd.DataFrame, direction: str) -> pd.DataFrame:
    """
    @ Todo
    """
    label_df = prepare_ttest_df(baseline_df, occlusion_df, "id")
    occlusion_df = occlusion_df.merge(preprocessing.ttest(label_df[["id", f"norm_explainability_score_{direction}"]],
                                                          label_df[
                                                              ["id", f"mean_norm_explainability_score_{direction}"]],
                                                          "norm_explainability_score", "id", 0.05), on="id")
    return occlusion_df.merge(preprocessing.ttest(label_df[["id", f"confidence_direction_{direction}"]],
                                                  label_df[["id", f"mean_confidence_direction_{direction}"]],
                                                  "confidence_direction", "id", 0.05), on="id")


def prepare_ttest_df(baseline_df: pd.DataFrame, occlusion_df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    @ Todo
    """
    baseline_df_mean = preprocessing.group_by_agg_column(baseline_df, col, agg_dict={"confidence_scaled": "mean"})
    baseline_df_mean = baseline_df_mean.rename({"confidence_scaled": "mean_confidence_direction_pos"}, axis=1)
    baseline_df_mean["mean_confidence_direction_neg"] = -baseline_df_mean["mean_confidence_direction_pos"]
    occlusion_df_p = occlusion_df[occlusion_df["confidence_direction"] > 0].groupby(col) \
        .agg({"confidence_scaled": list, "norm_explainability_score": list})
    occlusion_df_n = occlusion_df[occlusion_df["confidence_direction"] < 0].groupby(col) \
        .agg({"confidence_scaled": list, "norm_explainability_score": list})
    if not occlusion_df_n.empty and not occlusion_df_p.empty:
        baseline_df_mean = baseline_df_mean.merge(occlusion_df_p, on=col)
        baseline_df_mean = baseline_df_mean.merge(occlusion_df_n, on=col, suffixes=("_pos", "_neg"))
        baseline_df_mean = baseline_df_mean.rename({"confidence_scaled_pos": "confidence_direction_pos",
                                                    "confidence_scaled_neg": "confidence_direction_neg"}, axis=1)
        baseline_df_mean["confidence_direction_neg"] = baseline_df_mean["confidence_direction_neg"] \
            .apply(lambda lst: [-nr for nr in lst])
    if occlusion_df_n.empty:
        baseline_df_mean = baseline_df_mean.merge(occlusion_df_p, on=col)
        baseline_df_mean = baseline_df_mean.rename({"confidence_scaled": "confidence_direction_pos",
                                                    "norm_explainability_score": "norm_explainability_score_pos"},
                                                   axis=1)
    if occlusion_df_p.empty:
        baseline_df_mean = baseline_df_mean.merge(occlusion_df_n, on=col)
        baseline_df_mean = baseline_df_mean.rename({"confidence_scaled": "confidence_direction_neg",
                                                    "norm_explainability_score": "norm_explainability_score_neg"},
                                                   axis=1)
        baseline_df_mean["confidence_direction_neg"] = baseline_df_mean["confidence_direction_neg"] \
            .apply(lambda lst: [-nr for nr in lst])

    baseline_df_mean[["mean_norm_explainability_score_pos", "mean_norm_explainability_score_neg"]] = 0
    return baseline_df_mean


def annotation_analysis():
    """
    This functions prepares and performs the quantitative annotation analysis.
    Dumps individual core numbers into a json and Dataframes into a csv.
    Creates plots for quantitative analysis.
    """
    multilingual_mean_df_list = []
    for l in LANGUAGES:
        df_gold = preprocessing.get_accepted_cases(EXTRACTED_DATASETS_GOLD[f"annotations_{l}-{GOLD_SESSION}"])
        # Prepare json scores
        counts = {"total_number_of_cases": len(df_gold),
                  "number_of_comments": count_user_input(df_gold),
                  "number_of_not_accepted": count_not_accept(df_gold),
                  "number_of_is_correct": count_is_correct(df_gold)[0] / len(df_gold),
                  "number_of_approved_prediction": count_approved_dismissed(df_gold)[0] / len(df_gold),
                  "number_of_dismissed_prediction": count_approved_dismissed(df_gold)[1] / len(df_gold)
                  }
        preprocessing.write_json(f"{l}/quantitative/annotation_analysis_{l}.json", counts)
        # Preparation for IAA agreement plot

        # Prepare csv tables
        label_mean_df, pers_mean_df = get_annotation_label_mean(l), get_annotation_pers_mean(l)
        facts_count_df = count_mean_fact_length(df_gold)
        legal_area_count_df = get_legal_area_distribution(df_gold)
        preprocessing.write_csv_from_list(f"{l}/quantitative/annotation_analysis_{l}.csv",
                                          [label_mean_df, pers_mean_df,
                                           facts_count_df, legal_area_count_df])
        # Prepare plot tables
        pers_mean_df = pers_mean_df.drop(["label", "mean_token"], axis=1).fillna(0).groupby("annotator").sum()
        plots.ann_distribution_plot(facts_count_df.drop(f"mean_legal_area").drop(f"mean_year", axis=1),
                                    ax_labels=["Years", "Number of Tokens"],
                                    legend_texts=[la.replace("_", " ") for la in LEGAL_AREAS],
                                    ylim=[0, 600],
                                    title="Token over legal area and year Distribution",
                                    error_bars=pd.DataFrame(),
                                    mean_lines={},
                                    filepath=f"plots/ann_mean_tokens_year_legal_area_{l}.png")

        legend = [f"mean {label.lower()}" for label in LABELS_OCCLUSION[:-1]]
        plots.ann_distribution_plot(pers_mean_df[[f"{label.lower().replace(' ', '_')}_mean_token" for label in LABELS]],
                                    ax_labels=["Annotator", "Number of Tokens"],
                                    legend_texts=legend + [f"mean tokens {label.lower()}"
                                                 for label in LABELS_OCCLUSION[:-1]],
                                    ylim=[0, 140],
                                    title="Token Distribution of Annotation Labels",
                                    error_bars=pers_mean_df[[f"{label.lower().replace(' ', '_')}_error" for label in LABELS]],
                                    mean_lines={f"mean_tokens_{label.lower().replace(' ', '_')}":
                                          list(pers_mean_df[f"{label.lower().replace(' ', '_')}_mean"].values)[0]
                                      for label in LABELS},
                                    filepath=f"plots/ann_mean_tokens_exp_labels_{l}.png")

        multilingual_mean_df_list.append(label_mean_df.reset_index())
    # Mean token plot
    plots.multilingual_ann_plot(multilingual_mean_df_list)
    # IAA extraction Sets
    colors = [[COLORS["purple"], COLORS["dark green"], COLORS["dark blue"]],
              [COLORS["light purple"], COLORS["green"], COLORS["blue"]]]
    plots.violin_plot(1, 3, preprocessing.prepare_scores_IAA_Agreement_plots("de", [1, 2], "{}/{}_{}_{}.csv",
                                                                             ['annotations_de', 'tokens_text_angela',
                                                                              'tokens_text_lynn', 'tokens_text_thomas'],
                                                                             LABELS, IAA_List)[:-3],
                      ["A1 and A2", "A1 and A3", "A2 and A3"],
                      [[f"overlap_minimum_{label.lower().replace(' ', '_')}" for label in LABELS],
                       [f"rougeL_{label.lower().replace(' ', '_')}" for label in LABELS]],
                      legend_texts=["Overlap Minimum", "ROUGEL"],
                      title="Overlap Minimum/ROUGEL of Annotations",
                      filepath="plots/ann_o_min_rL_.png", legendpath="plots/ann_oo_min_rL__legend_{}.png",
                      annotation='Agreement between {}', colors=colors)
    plots.violin_plot(1, 3, preprocessing.prepare_scores_IAA_Agreement_plots("de", [1, 2], "{}/{}_{}_{}.csv",
                                                                             ['annotations_de', 'tokens_text_angela',
                                                                              'tokens_text_lynn', 'tokens_text_thomas'],
                                                                             LABELS, IAA_List)[:-3],
                      ["A1 and A2", "A1 and A3", "A2 and A3"]
                      , [[f"bert_score_{label.lower().replace(' ', '_')}" for label in LABELS],
                         [f"meteor_score_{label.lower().replace(' ', '_')}" for label in LABELS]],
                      legend_texts=["Bert Score", "METEOR Score"],
                      title="BERTScore/METEOR Score Score of Annotations",
                      filepath="plots/ann_be_me.png", legendpath="plots/ann_o_be_me_legend_{}.png",
                      annotation='Agreement between {}', colors=colors)
    preprocessing.write_IAA_table_ann(preprocessing.prepare_scores_IAA_Agreement_plots("de", [1, 2], "{}/{}_{}_{}.csv",
                                                                                       ['annotations_de',
                                                                                    'tokens_text_angela',
                                                                                    'tokens_text_lynn',
                                                                                    'tokens_text_thomas'], LABELS,
                                                                                       IAA_List)[:-3],
                                  "tables/ann_IAA_1{}.csv", LABELS)
    preprocessing.write_IAA_table_ann(preprocessing.prepare_scores_IAA_Agreement_plots("de", [1, 2], "{}/{}_{}_{}.csv",
                                                                                       ['annotations_de',
                                                                                    'tokens_text_angela',
                                                                                    'tokens_text_lynn',
                                                                                    'tokens_text_thomas'], LABELS,
                                                                                       IAA_List)[3:],
                                  "tables/ann_IAA_2{}.csv", LABELS)


def lower_court_analysis():
    """
    This functions prepares and performs the quantitative lower_court analysis.
    Dumps individual core numbers into a json.
    Creates plots for quantitative analysis.
    """
    lower_court_df_dict = {l: [] for l in LANGUAGES}
    for l in LANGUAGES:
        lower_court_df = preprocessing.read_csv(OCCLUSION_PATHS["analysis"][0].format(l, l), "index")
        lower_court_df["lower_court_long"] = lower_court_df["lower_court"].copy()
        lower_court_df["lower_court"] = lower_court_df["lower_court"].apply(lambda row: abbreviate_lower_courts(l, row))
        baseline_df = preprocessing.get_baseline(lower_court_df)
        lower_court_df = preprocessing.remove_baseline(lower_court_df).sort_values("lower_court")
        # Prepare json scores
        lower_court_dict = {"number_of_distinct_lower_court": len(list(baseline_df["lower_court"].unique())),
                            "number_of_full_lower_courts": count_lower_court(baseline_df)[0],
                            "full_lower_court_list": list(baseline_df["lower_court_long"].unique()),
                            "mean_tokens_full_lower_court": count_lower_court(baseline_df)[1],
                            "number_of_has_flipped_lc(T/F/Total)": list(count_has_flipped(lower_court_df)),
                            "number_of_approved_prediction_baseline": count_approved_dismissed(baseline_df)[0] / len(
                               baseline_df),
                            "number_of_dismissed_prediction_baseline": count_approved_dismissed(baseline_df)[1] / len(
                                baseline_df),
                            "number_of_approved_prediction": count_approved_dismissed(lower_court_df)[0] / len(lower_court_df),
                            "number_of_dismissed_prediction": count_approved_dismissed(lower_court_df)[1] / len(lower_court_df),

                            }
        preprocessing.write_json(f"{l}/quantitative/lower_court_analysis_{l}.json", lower_court_dict)

        # Lower court has flipped distribution plots
        plots.create_lc_flipped_plot(l, lower_court_df, ["lower_court", "has_not_flipped", "has_flipped"],
                                     label_texts=["Lower Court", "Ratio of Experiments"],
                                     legend_texts=[f"{lst[0]}flipped Prediction {lst[1]}" for lst in
                                                            [["not ", 0], ["", 0], ["not ", 1], ["", 1]]],
                                     title=f"Distribution of flipped Experiments per Lower Court",
                                     filepath=f'plots/lc_flipped_distribution_{l}.png')

        # Lower court legal area distribution plots
        legal_area_distribution, legal_area_conf = get_lc_la_distribution(
            lower_court_df.drop(["norm_explainability_score", "confidence_direction"], axis=1))
        distribution_df = get_lc_distribution(lower_court_df)
        preprocessing.write_csv(f"tables/lc_distribution_{l}.csv",
                                distribution_df.reset_index())
        # lower_court legal_area distribution plot
        legal_area_distribution[LEGAL_AREAS] = legal_area_distribution[LEGAL_AREAS].mul(distribution_df["count"], axis=0)
        lc_la_df = legal_area_distribution.set_index("lower_court").T
        plots.distribution_plot(len(lc_la_df.columns),
                                lc_la_df.reindex(sorted(lc_la_df.columns), axis=1), rows=LEGAL_AREAS,
                                ax_labels=["Lower Court", "Ratio of Experiments"],
                                x_labels=[i + 1 for i in range(0, len(sorted(list(lc_la_df.columns))))],
                                legend_texts=tuple([la.replace("_", " ") for la in LEGAL_AREAS]),
                                ylim=[],
                                errorbars=pd.DataFrame(),
                                title=f"Lower Court distribution over Legal Area",
                                filepath=f'plots/lc_distribution_la_{l}.png')



        # Lower court distribution table



        # Preparation for effect plots
        lower_court_df_dict[l].append(lower_court_df)
        lower_court_df_dict[f"{l}_mu"] = prepare_ttest_df(baseline_df, lower_court_df, "lower_court")

    # Lower court effect plots
    plots.create_lc_effect_plot(lower_court_df_dict,
                                cols=["lower_court", "mean_norm_explainability_score"],
                                label_texts=["Explainability Score", "Lower Court"],
                                legend_texts=["Prediction 0: Exp_Score > 0","Prediction 1: Exp_Score > 0",
                                           "Prediction 0: Exp_Score >> 0", "Prediction 1: Exp_Score >> 0",
                                           "Prediction 0: Exp_Score < 0", "Prediction 1: Exp_Score < 0",
                                           "Prediction 0: Exp_Score << 0", "Prediction 1: Exp_Score << 0"],
                                xlim=[-0.09, 0.09],
                                title="Mean Explainability Scores in both directions (with Significance) {}",
                                filepath='plots/lc_effect_mean_{}.png')


def occlusion_analysis():
    """
    @Todo Comment & clean up
    Very important note!!!!!!! Only cases from test set were accepted in occlusion!
    """
    scatter_plot_dict = {l: {'c_o': [], 'c_s': [], 'f_o': [], 'f_s': [], 'o': [], 's': []} for l in LANGUAGES}

    multilingual_mean_length_dict = {}
    for nr in NUMBER_OF_EXP:
        prediction_dict = {l: {} for l in LANGUAGES}
        multilingual_mean_length_dict[nr] = []
        for l in LANGUAGES:
            occlusion_df = preprocessing.read_csv(OCCLUSION_PATHS["analysis"][1].format(l, nr, l), "index")
            baseline_df = preprocessing.get_baseline(occlusion_df)
            occlusion_df = preprocessing.remove_baseline(occlusion_df)

            # Append to multilingual lists
            multilingual_mean_length_dict[nr].append(get_occlusion_label_mean(occlusion_df)[1])

            occlusion_df = occlusion_df[
                occlusion_df["confidence_direction"] != 0]  # ignore neutral classification
            occlusion_df, false_classification, correct_classification = preprocessing.get_correct_direction(
                occlusion_df)
            score_0, score_1 = false_classification.groupby("explainability_label")["correct_direction"].count(), \
                               correct_classification.groupby("explainability_label")["correct_direction"].count()

            # Prepare json scores
            occlusion_dict = {"number of distinct cases": len(list(baseline_df["id"].values)),
                              "mean sentence length": get_occlusion_label_mean(occlusion_df)[0],
                              "mean number of sentences": get_number_of_sentences(occlusion_df),
                              "number of correct classification": score_1.to_dict(),
                              "number of incorrect_classification": score_0.to_dict()}
            prediction_dict[l] = {"number_of_approved_prediction_baseline": count_approved_dismissed(baseline_df)[0] / len(
                                  baseline_df),
                              "number_of_dismissed_prediction_baseline": count_approved_dismissed(baseline_df)[1] / len(
                                  baseline_df),
                              "number_of_approved_prediction": count_approved_dismissed(occlusion_df)[0] / len(
                                  occlusion_df),
                              "number_of_dismissed_prediction": count_approved_dismissed(occlusion_df)[1] / len(
                                 occlusion_df)}
            preprocessing.write_json(f"{l}/quantitative/occlusion_analysis_{l}_{nr}.json", occlusion_dict)

            o_judgement_f, s_judgement_f = split_oppose_support_false(occlusion_df, false_classification)
            o_judgement_c, s_judgement_c = split_oppose_support_correct(correct_classification)
            # IAA scores calculation
            # Keep this commented except if running on Ubelix
            #scores.write_IAA_to_csv_occlusion(s_judgement_f, l, f"{l}/occ_supports_judgment_{l}_{nr}.csv")
            #scores.write_IAA_to_csv_occlusion(o_judgement_f, l, f"{l}/occ_opposes_judgment_{l}_{nr}.csv")
            preprocessing.write_csv(f"{l}/c_occ_opposes_judgment_{l}_{nr}.csv", o_judgement_c)
            preprocessing.write_csv(f"{l}/c_occ_supports_judgment_{l}_{nr}.csv", s_judgement_c)


            # Occlusion effect plots preparation
            o_judgement_f, s_judgement_f = apply_occ_ttest(baseline_df, o_judgement_f, "pos"), \
                                           apply_occ_ttest(baseline_df, s_judgement_f, "neg")
            o_judgement_c, s_judgement_c = apply_occ_ttest(baseline_df, o_judgement_c, "pos"), \
                                           apply_occ_ttest(baseline_df, s_judgement_c, "neg")

            s_judgement_f["confidence_scaled"], s_judgement_c["confidence_scaled"] = s_judgement_f["confidence_scaled"]*-1, s_judgement_c["confidence_scaled"]*-1
            scatter_plot_dict[l]['c_o'].append(o_judgement_c)
            scatter_plot_dict[l]['c_s'].append(s_judgement_c)
            scatter_plot_dict[l]['f_o'].append(o_judgement_f)
            scatter_plot_dict[l]['f_s'].append(s_judgement_f)
            occlusion_df = pd.concat([o_judgement_f, s_judgement_f, o_judgement_c, s_judgement_c])
            s_judgement, o_judgement = occlusion_df[occlusion_df["numeric_label_human"] == -1], \
                                       occlusion_df[occlusion_df["numeric_label_human"] == 1]
            scatter_plot_dict[l]["s"].append(s_judgement)
            scatter_plot_dict[l]["o"].append(o_judgement)
        # Prediction table
        prediction_df = pd.DataFrame.from_dict(prediction_dict).T
        preprocessing.write_csv(f"tables/prediction_distribution_{nr}.csv", prediction_df)
    # Scatter plot
    plots.create_occ_scatter_plot(scatter_plot_dict)
    # Occlusion mean token distribution
    plots.create_multilingual_occ_plot(multilingual_mean_length_dict)
    colors = [[COLORS["dark green"], COLORS["dark blue"]], [COLORS["green"], COLORS["blue"]]]
    for l in LANGUAGES:
        IAA_df_list = preprocessing.prepare_scores_IAA_Agreement_plots(l, NUMBER_OF_EXP,
                                                                       "{}/occlusion/occ_{}_{}_{}.csv", [],
                                                                       LABELS[1:], ["human_model"])
        plots.violin_plot(2, 2, IAA_df_list,
                          [str(i) for i in range(0, 4)],
                          [[f"bert_score_{label.lower().replace(' ', '_')}" for label in LABELS[1:]],
                           []],
                          legend_texts=["BERTScore"],
                          title=f"BERTScore between Human and Model {l.upper()}",
                          filepath=f"plots/occ_be_{l}.png", legendpath="plots/occ_o_be_legend_{}.png",
                          annotation="Occlusion with {} Sentences", colors=colors[:-1])
        preprocessing.write_IAA_table_occ(IAA_df_list, f"tables/occ_IAA_{l}{{}}.csv", LABELS[1:])
