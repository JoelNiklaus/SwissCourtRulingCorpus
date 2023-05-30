import numpy as np
import pandas as pd

import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing
from scrc.annotation.judgment_explainability.analysis.utils import plots

"""
This workflow produces the explanations for each experiment size (1 sentence, 2 sentence etc.) 
with one sentence speaking most for the judgment and one most against the judgment 
according to the two flavors model-near/human-near.
"""


LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
LANGUAGES = ["de", "fr", "it"]
NUMBER_OF_EXP = [1, 2, 3, 4]
COLUMNS = ['id', 'prediction', 'year', "legal_area", 'label', 'confidence_direction', 'norm_explainability_score',
           'correct_direction',
           'explainability_label_model',
           'occluded_text_model']


def find_indexes(lst_1: list, lst_2: list, label) -> list:
    """
    Find indexes according to conditions, appends list entries according to indexes.
    Returns list of selected entries.
    """
    result_list = []
    if label == LABELS[1]:
        search_indexes = list(np.where(np.array(lst_1) == min(lst_1))[0])
    if label == LABELS[2]:
        search_indexes = list(np.where(np.array(lst_1) == max(lst_1))[0])
    for i in search_indexes:
        result_list.append(lst_2[i])
    return result_list


def apply_list_mean(row_1, row_2, label) -> list:
    """
    Returns mean of list list containing correct entries.
    """
    return [sum(find_indexes(row_2, row_1, label)) / len(find_indexes(row_2, row_1, label))]


def model_agg(label_df: pd.DataFrame, col, label: str, label_agg: str) -> pd.DataFrame:
    """
    Performs selection of model-near explanations, by grouping by id and aggregating the needed columns to a list.
    Select the explanation with the highest "norm_explainability_score".
    Returns exploded Dataframe containing the model-nearest explanations index and score.
    """
    ex_score = label_df[["index", f"id_{label.lower().replace(' ', '_')}", col,
                         f"bert_score_{label.lower().replace(' ', '_')}"]].groupby(
        f"id_{label.lower().replace(' ', '_')}") \
        .agg({"index": list, col: list, f"bert_score_{label.lower().replace(' ', '_')}": list})
    ex_score["index"] = ex_score.apply(lambda row: find_indexes(row[col], row["index"], label_agg), axis=1)
    ex_score["score"] = ex_score.apply(
        lambda row: find_indexes(row[col], row[col], label_agg), axis=1)
    ex_score[f"bert_score_{label.lower().replace(' ', '_')}"] = ex_score.apply(
        lambda row: apply_list_mean(row[f"bert_score_{label.lower().replace(' ', '_')}"], row[col], label_agg),
        axis=1)
    ex_score["score"] = ex_score["score"].apply(lambda row: [row[0]])
    ex_score["index"] = ex_score["index"].apply(lambda row: [row[0]])
    ex_score["score"] = ex_score[f"bert_score_{label.lower().replace(' ', '_')}"]
    return ex_score


def human_agg(label_df: pd.DataFrame, col, label: str, label_agg: str):
    """
    Performs selection of human-near explanations, by grouping by id and aggregating the needed columns to a list.
    Select the explanation with the highest "bert_score".
    Returns exploded Dataframe containing the human-nearest explanations index and score.
    """
    ex_score = label_df[["index", f"id_{label.lower().replace(' ', '_')}", col]].groupby(
        f"id_{label.lower().replace(' ', '_')}") \
        .agg({"index": list, col: list})
    ex_score["index"] = ex_score.apply(lambda row: find_indexes(row[col], row["index"],label_agg), axis=1)
    ex_score["score"] = ex_score.apply(
        lambda row: find_indexes(row[col], row[col], label_agg), axis=1)
    ex_score["score"] = ex_score["score"].apply(lambda row: [row[0]])
    ex_score["index"] = ex_score["index"].apply(lambda row: [row[0]])
    return ex_score


def produce_model_human_explanation(label_df: pd.DataFrame, col, label: str, label_agg: str, mode: str):
    """
    Applies either model_agg or human_agg and explodes list columns.
    Returns merged Dataframe containing the selected explanations.
    """
    if mode == "model":
        ex_score = model_agg(label_df, col, label, label_agg)
    if mode == "human":
        ex_score = human_agg(label_df, col, label, label_agg)
    # Explodes list columns
    ex_score = ex_score[["index", "score"]].explode(["index", "score"])
    label_df = label_df[
        ['index', f"id_{label.lower().replace(' ', '_')}", f"prediction_{label.lower().replace(' ', '_')}",
         f"legal_area_{label.lower().replace(' ', '_')}",
         f"year_{label.lower().replace(' ', '_')}", f"occluded_text_model_{label.lower().replace(' ', '_')}",
         ]] \
        .merge(ex_score, on="index", how="inner")
    return label_df


def join_df(df_1, df_2) -> pd.DataFrame:
    """
    Renames the "id_label.lower().replace(' ', '_')" and "score" columns in each DataFrame
    Concatenates the two DataFrames and drops unnecessary columns.
    Returns merged Dataframe removes any duplicate rows, and resets the index and sorts by id.
    """
    text_score_1, text_score_2 = df_1[[f"id_{LABELS[1].lower().replace(' ', '_')}",
                                       f"occluded_text_model_{LABELS[1].lower().replace(' ', '_')}", "score"]], \
                                 df_2[[f"id_{LABELS[2].lower().replace(' ', '_')}",
                                       f"occluded_text_model_{LABELS[2].lower().replace(' ', '_')}", "score"]],
    text_score_1 = text_score_1.rename(
        {f"id_{LABELS[1].lower().replace(' ', '_')}": "id", "score": f"score_{LABELS[1].lower().replace(' ', '_')}"},
        axis=1)
    text_score_2 = text_score_2.rename({f"id_{LABELS[2].lower().replace(' ', '_')}": "id",
                                        "score": f"score_{LABELS[2].lower().replace(' ', '_')}"}, axis=1)
    df_1.columns, df_2.columns = df_1.columns.str.replace(f"_{LABELS[1].lower().replace(' ', '_')}", ""),\
                                 df_2.columns.str.replace(f"_{LABELS[2].lower().replace(' ', '_')}", "")
    try:
        df = pd.concat([df_1, df_2]).drop(["occluded_text_model", "score", "bert_score"], axis=1).drop_duplicates()
    except KeyError:
        df = pd.concat([df_1, df_2]).drop(["occluded_text_model", "score"], axis=1).drop_duplicates()

    df = df.merge(text_score_1, on="id", how="outer").merge(text_score_2, on="id", how="outer").drop(["index"],
                                                                                                     axis=1).drop_duplicates()
    df["score"] = (df[f"score_{LABELS[1].lower().replace(' ', '_')}"].fillna(0) + df[
        f"score_{LABELS[2].lower().replace(' ', '_')}"].fillna(0))
    df["mean_score"] = df["score"] / 2
    return df.sort_values(by="id").drop_duplicates().reset_index()


def hist_preprocessing(df_list):
    """
    Prepares df_list for histogram by grouping by "id" and applying mean.
    Returns merged Dataframe dropping all duplicates.
    """
    i = 1
    hist_df = df_list[0][["id", "score"]].groupby("id").mean().reset_index().rename({"score": f"score_{0}"}, axis=1)
    for exp_df in df_list[1:]:
        exp_df = exp_df[["id", "score"]].groupby("id").mean().reset_index().rename({"score": f"score_{i}"}, axis=1)
        hist_df = hist_df.merge(exp_df, on="id")
        i += 1
    return hist_df.drop_duplicates()


def prepare_json(df_list) -> dict:
    """
    Returns dict with values for json dump.
    """
    json_dict = {nr: {} for nr in range(1, 5)}
    i = 1
    for df in df_list:
        json_dict[i]["mean_score"] = df["mean_score"].mean()
        json_dict[i]["opposes_judgment"] = df["score_opposes_judgment"].mean()
        json_dict[i]["supports_judgment"] = df["score_supports_judgment"].mean()
        json_dict[i]["prediction"] = df.groupby("prediction")["mean_score"].mean().to_dict()
        json_dict[i]["legal_area"] = df.groupby("legal_area")["mean_score"].mean().to_dict()
        json_dict[i]["year"] = df.groupby("year")["mean_score"].mean().to_dict()
        i += 1
    return json_dict


def produce_explanation():
    """
    Extracts all needed Datasets.
    For each experiment, label and flavor appends Dataframe containing selected explanations.
    Creates histogram for explanation accuracy score distribution.
    Saves text explanations in csv and mean values as json.
    """

    for l in LANGUAGES:
        df_list_1 = preprocessing.prepare_scores_IAA_Agreement_plots(l, NUMBER_OF_EXP,
                                                                     "../{}/occlusion/occ_{}_{}_{}.csv",
                                                                     [], LABELS[1:], ["human_model"])
        df_list_2 = []
        for i in range(1, 5):
            df_list_2.append(
                pd.concat([preprocessing.read_csv(f"../{l}/c_occ_supports_judgment_{l}_{i}.csv", "index").reset_index(),
                           preprocessing.read_csv(f"../{l}/c_occ_opposes_judgment_{l}_{i}.csv",
                                                  "index").reset_index()]))
        label_df_model_list = []
        label_df_human_list = []
        for df_1, df_2 in zip(df_list_1, df_list_2):
            for label in LABELS[1:]:
                correct_df = df_2[df_2["explainability_label_model"] == label].drop(["index"], axis=1)
                correct_df = correct_df[COLUMNS].add_suffix(f"_{label.lower().replace(' ', '_')}").reset_index()
                incorrect_df = df_1[
                    ["index", f"bert_score_{label.lower().replace(' ', '_')}"] + list(correct_df.columns)[1:]]
                label_df = pd.concat([incorrect_df, correct_df], ignore_index=True).drop(["index"],
                                                                                         axis=1).reset_index().fillna(1)
                model_df = produce_model_human_explanation(label_df,
                                                           f"norm_explainability_score_{label.lower().replace(' ', '_')}",
                                                           label, label, "model")
                label_df_model_list.append(model_df.drop(["index"], axis=1).drop_duplicates().reset_index())
                human_df = produce_model_human_explanation(label_df, f"bert_score_{label.lower().replace(' ', '_')}",
                                                           label,
                                                           LABELS[2], "human")
                label_df_human_list.append(human_df.drop(["index"], axis=1).drop_duplicates().reset_index())

        model_list = [join_df(label_df_model_list[0], label_df_model_list[1]),
                      join_df(label_df_model_list[2], label_df_model_list[3]),
                      join_df(label_df_model_list[4], label_df_model_list[5]),
                      join_df(label_df_model_list[6], label_df_model_list[7])]
        human_list = [join_df(label_df_human_list[0], label_df_human_list[1]),
                      join_df(label_df_human_list[2], label_df_human_list[3]),
                      join_df(label_df_human_list[4], label_df_human_list[5]),
                      join_df(label_df_human_list[6], label_df_human_list[7])]

        plots.explanation_histogram(l, hist_preprocessing(model_list), hist_preprocessing(human_list),
                                    f"../plots/occ_explanation_hist_{l}.png")
        preprocessing.write_csv_from_list(f"../tables/model_explanation_{l}.csv", model_list)
        preprocessing.write_csv_from_list(f"../tables/human_explanation_{l}.csv", human_list)
        preprocessing.write_json(f"../{l}/qualitative/explanation_mean.json", {"model": prepare_json(model_list),
                                                                               "human": prepare_json(human_list)})


produce_explanation()
