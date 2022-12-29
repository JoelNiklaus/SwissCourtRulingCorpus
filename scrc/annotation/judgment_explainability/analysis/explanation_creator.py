import pandas as pd
from matplotlib import pyplot as plt

import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing
from scrc.annotation.judgment_explainability.analysis.utils import plots

LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
LANGUAGES = ["de", "fr", "it"]
NUMBER_OF_EXP = [1, 2, 3, 4]
COLUMNS = ['id', 'prediction', 'year', 'label', 'confidence_direction', 'norm_explainability_score', 'correct_direction',
       'explainability_label_model',
       'occluded_text_model']


def apply_label_agg_1(row_1, row_2, label):
    if label == LABELS[1]:
        return row_1[row_2.index(min(row_2))]
    if label == LABELS[2]:
        return row_1[row_2.index(max(row_2))]


def apply_label_agg_2(row_1, label):
    if label == LABELS[1]:
        return min(row_1)
    if label == LABELS[2]:
        return max(row_1)


def model_agg(label_df: pd.DataFrame, col, label: str, label_agg: str):
    ex_score = label_df[["index", f"id_{label.lower().replace(' ', '_')}", col,
                         f"bert_score_{label.lower().replace(' ', '_')}"]].groupby(
        f"id_{label.lower().replace(' ', '_')}") \
        .agg({"index": list, col: list, f"bert_score_{label.lower().replace(' ', '_')}":list})
    ex_score["index"] = ex_score.apply(lambda row: apply_label_agg_1(row["index"], row[col], label_agg), axis=1)
    ex_score[f"bert_score_{label.lower().replace(' ', '_')}"] = ex_score.apply(lambda row: apply_label_agg_1(row[f"bert_score_{label.lower().replace(' ', '_')}"], row[col], label_agg), axis=1)
    ex_score["score"] = ex_score.apply(
        lambda row: apply_label_agg_2(row[col], label_agg), axis=1)
    return ex_score[["index", "score", f"bert_score_{label.lower().replace(' ', '_')}"]]


def human_agg(label_df: pd.DataFrame, col, label: str, label_agg: str):
    ex_score = label_df[["index", f"id_{label.lower().replace(' ', '_')}", col]].groupby(
        f"id_{label.lower().replace(' ', '_')}") \
        .agg({"index": list, col: list})
    ex_score["index"] = ex_score.apply(lambda row: apply_label_agg_1(row["index"], row[col], label_agg), axis=1)

    ex_score["score"] = ex_score.apply(
        lambda row: apply_label_agg_2(row[col], label_agg), axis=1)
    return ex_score[["index", "score"]]


def produce_model_human_explanation(label_df: pd.DataFrame, col, label: str, label_agg: str, mode: str):
    if mode == "human":
        ex_score = human_agg(label_df, col, label, label_agg)
    if mode == "model":
        ex_score = model_agg(label_df, col, label, label_agg)
    label_df = label_df[
        ['index', f"id_{label.lower().replace(' ', '_')}", f"prediction_{label.lower().replace(' ', '_')}",
         f"year_{label.lower().replace(' ', '_')}", f"occluded_text_model_{label.lower().replace(' ', '_')}",
         ]] \
        .merge(ex_score, on="index", how="inner")
    return label_df


def join_df(df_1, df_2):
    text_score_1, text_score_2 = df_1[[f"id_{LABELS[1].lower().replace(' ', '_')}",f"occluded_text_model_{LABELS[1].lower().replace(' ', '_')}", "score"]], \
                                df_2[[f"id_{LABELS[2].lower().replace(' ', '_')}",f"occluded_text_model_{LABELS[2].lower().replace(' ', '_')}", "score"]],
    text_score_1, text_score_2 = text_score_1.rename({f"id_{LABELS[1].lower().replace(' ', '_')}":"id", "score": f"score_{LABELS[1].lower().replace(' ', '_')}"}, axis=1),\
                                 text_score_2.rename({f"id_{LABELS[2].lower().replace(' ', '_')}":"id", "score": f"score_{LABELS[2].lower().replace(' ', '_')}"}, axis=1)
    df_1.columns, df_2.columns = df_1.columns.str.replace(f"_{LABELS[1].lower().replace(' ', '_')}", ""), \
                                 df_2.columns.str.replace(f"_{LABELS[2].lower().replace(' ', '_')}", "")
    try:
        df = pd.concat([df_1, df_2]).drop(["occluded_text_model", "score", "bert_score"], axis=1).drop_duplicates()
    except KeyError:
        df = pd.concat([df_1, df_2]).drop(["occluded_text_model", "score"], axis=1).drop_duplicates()

    df = df.merge(text_score_1, on="id", how="outer").merge(text_score_2, on="id", how="outer").drop(["index"], axis=1).drop_duplicates()
    df["score"] = (df[f"score_{LABELS[1].lower().replace(' ', '_')}"].fillna(0) + df[f"score_{LABELS[2].lower().replace(' ', '_')}"].fillna(0))
    df["mean_score"] = df["score"] / 2
    return df.sort_values(by="id").reset_index()


def hist_preprocessing(df_list):
    i = 1
    hist_df = df_list[0][["id", "score"]].groupby("id").mean().reset_index().rename({"score": f"score_{0}"}, axis=1)
    for exp_df in df_list[1:]:
        exp_df = exp_df[["id", "score"]].groupby("id").mean().reset_index().rename({"score": f"score_{i}"}, axis=1)
        hist_df = hist_df.merge(exp_df, on="id")
        i += 1
    return hist_df


def produce_explanation():
    """
    @Todo
    For each experiment size (1 sentence, 2 sentence etc.)
    Produce the sentence speaking most for the judgemnt and most against the judgement
    These are either the sentences which were correctly interpreted by bert or those with the highest agreement.
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
                incorrect_df = df_1[["index", f"bert_score_{label.lower().replace(' ', '_')}"] + list(correct_df.columns)[1:]]
                label_df = pd.concat([incorrect_df,correct_df], ignore_index=True).drop(["index"], axis=1).reset_index().fillna(1)
                model_df = produce_model_human_explanation(label_df,
                                                           f"norm_explainability_score_{label.lower().replace(' ', '_')}",
                                                           label, label, "model")
                model_df["score"] = model_df[f"bert_score_{label.lower().replace(' ', '_')}"].fillna(1)
                label_df_model_list.append(model_df)
                human_df = produce_model_human_explanation(label_df, f"bert_score_{label.lower().replace(' ', '_')}", label,
                                                    LABELS[2], "human")
                label_df_human_list.append(human_df)

        model_list = [join_df(label_df_model_list[0], label_df_model_list[1]),
                      join_df(label_df_model_list[2], label_df_model_list[3]),
                      join_df(label_df_model_list[4], label_df_model_list[5]),
                      join_df(label_df_model_list[6], label_df_model_list[7])]
        human_list = [join_df(label_df_human_list[0], label_df_human_list[1]),
                      join_df(label_df_human_list[2], label_df_human_list[3]),
                      join_df(label_df_human_list[4], label_df_human_list[5]),
                      join_df(label_df_human_list[6], label_df_human_list[7])]



        plots.explanation_histogram(hist_preprocessing(model_list), hist_preprocessing(human_list),f"../plots/occ_explanation_hist_{l}.png")
        preprocessing.write_csv_from_list(f"../tables/model_explanation_{l}.csv", model_list)
        preprocessing.write_csv_from_list(f"../tables/human_explanation_{l}.csv", human_list)
        json_dict = {"model": {nr: "" for nr in range(1, 5)},
                     "human": {nr: "" for nr in range(1, 5)}}
        i = 1
        for df_1, df_2 in zip(model_list, human_list):
            json_dict["model"][i] = df_1["mean_score"].mean()
            json_dict["human"][i] = df_2["mean_score"].mean()
            i += 1

        preprocessing.write_json(f"../{l}/qualitative/explanation_mean.json", json_dict)


produce_explanation()
