import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

import scrc.annotation.judgment_explainability.analysis.utils.plots as plots
import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing
import scrc.annotation.judgment_explainability.analysis.utils.scores as scores

warnings.filterwarnings("ignore")

"""
Contains functions for the quantitative analysis. Uses preprocessing.py and plots.py. 
Is used by annotation_analysis and occlusion_analysis.
"""
LANGUAGES = ["de", "fr", "it"]
PERSONS = ["angela", "lynn", "thomas"]
LABELS_OCCLUSION = ["Lower court", "Supports judgment", "Opposes judgment", "Neutral"]
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
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
                   "analysis": ("{}/occlusion/bias_lower_court_{}.csv", "{}/occlusion/occlusion_{}_{}.csv")}

NUMBER_OF_EXP = [1, 2, 3, 4]


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
    return len(dataset[dataset["is_correct"] is True].groupby("id_csv")), len(dataset)


def count_approved_dismissed(dataset: pd.DataFrame) -> (int, int):
    """
    Returns number approved cases and number of dismissed cases.
    """
    return len(dataset[dataset["judgment"] == "approval"].groupby("id_csv")), \
           len(dataset[dataset["judgment"] == "dismissal"].groupby("id_csv"))


def count_lower_court(dataset: pd.DataFrame):
    """
    Returns number of distinct lower courts and their length in tokens.
    """
    dataset["length"] = dataset["lower_court"].apply(lambda x: len(word_tokenize(x)))
    return dataset["lower_court"].nunique(), dataset["length"].mean()


def count_has_flipped(dataset: pd.DataFrame) -> (int, int, int):
    """
    Returns number of flipped experiments, number of unflipped experiments and length of Dataframe.
    """
    return len(dataset[dataset["has_flipped"] == True]), len(dataset[dataset["has_flipped"] == False]), len(dataset)


def group_by_agg_column(dataset: pd.DataFrame, col_1, agg_dict: dict) -> pd.DataFrame:
    """
    Groups by col_1 and applies count of "id" and mean of 'confidence_scaled'.
    Returns Dataframe
    """
    return dataset.groupby(col_1).agg(agg_dict).reset_index()


def get_label_mean_df(lang: str) -> pd.DataFrame:
    """
    Gets number of tokens of each explainability label.
    Calculates mean token number of each explainability label.
    Returns Dataframe.
    """
    label_df_list = []
    label_mean = {"label": [], "mean_token": []}
    for label in LABELS_OCCLUSION:
        try:
            label_df = preprocessing.read_csv(f"{lang}/{label.lower().replace(' ', '_')}_{lang}_4.csv", "index")
            label_df["length"] = label_df["tokens_id"].apply(lambda x: get_length(x))
            label_df_list = label_df_list + list(label_df[f"length"].values)
            label_mean["label"].append(label)
            label_mean["mean_token"].append(
                float(sum(label_df_list) / len(list(filter(lambda num: num != 0, label_df_list)))))
        except KeyError:
            pass
    return pd.DataFrame.from_dict(label_mean)


def get_pers_mean_df(lang: str) -> pd.DataFrame:
    """
    Gets number of tokens of each explainability label per person and case.
    Calculates mean token number of each explainability label per person.
    Returns Dataframe.
    """
    pers_mean = {"label": [], "annotator": [], "mean_token": []}
    for label in LABELS:
        label_df = preprocessing.read_csv(f"{lang}/{label.lower().replace(' ', '_')}_{lang}_3.csv", "index")
        pers_mean[f"{label.lower().replace(' ', '_')}_mean_token"] = []
        for person in PERSONS:
            try:
                label_df[f"length_{person}"] = label_df[f"tokens_id_{person}"].apply(lambda x: get_length(x))
                pers_mean["label"].append(label)
                pers_mean["annotator"].append(person)
                pers_mean["mean_token"].append(label_df[f"length_{person}"].mean())
                pers_mean[f"{label.lower().replace(' ', '_')}_mean_token"].append(
                    label_df[f"length_{person}"].mean())
            except KeyError:
                pass
        return pd.DataFrame.from_dict(dict([(k, pd.Series(v)) for k, v in pers_mean.items()]))


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
    Calculates proportional distibution of cases per legal area.
    Returns Dataframe.
    """
    distribution_df = dataset.groupby("legal_area").count().reset_index()[["legal_area", "id_scrc"]]
    distribution_df["distribution"] = distribution_df["id_scrc"].div(len(dataset))
    return distribution_df.drop("id_scrc", axis=1, inplace=False)


def get_lower_court_distribution(dataset: pd.DataFrame) -> (pd.DataFrame):
    """
    Sorts dataset according to normal distribution
    Counts proportional occurrences of each lower court.
    Returns Dataframe
    """
    distribution_df = sort_normal_distribution(dataset.groupby("lower_court")["id"].count().reset_index()) \
        .reset_index().rename(columns={"index": "lower_court", 0: "count"})
    distribution_df["count"] = distribution_df["count"].div(distribution_df["count"].sum())
    return distribution_df


def sort_normal_distribution(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts according to normal distribution (minimums at beginning and ends).
    Returns sorted Dataframe.
    """
    df_list = dataset.values.tolist()
    df_dict = {item[0]: item[1:][0] for item in df_list}
    result_list = [len(df_dict) * [None], len(df_dict) * [None]]
    i = 0
    while len(df_dict) > 0:
        result_list[0][-(1 + i)] = min(df_dict.values())
        result_list[1][-(1 + i)] = min(df_dict, key=df_dict.get)
        del df_dict[min(df_dict, key=df_dict.get)]
        result_list[0][i] = min(df_dict.values())
        result_list[1][i] = min(df_dict, key=df_dict.get)
        del df_dict[min(df_dict, key=df_dict.get)]
        i += 1
    return pd.DataFrame(result_list[0], index=result_list[1])


def get_one_sided_agg(dataset: pd.DataFrame, sorted_df: pd.DataFrame):
    """
    Gets mean of one sided explainability_score and confidence_direction.
    Returns Dataframe.
    """
    dataset = group_by_agg_column(dataset, "lower_court",
                                  agg_dict={"confidence_direction": "mean", "norm_explainability_score": "mean"}) \
        .rename(columns={"norm_explainability_score": "mean_norm_explainability_score"})
    return sorted_df.merge(dataset, on="lower_court", how="inner")


def get_lc_la_distribution(dataset: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Groups by lower_court and applies list() to rows.
    Gets mean confidence_scaled per legal_area and distribution over legal areas.
    Returns Dataframes.
    """
    legal_area_df = group_by_agg_column(dataset, "lower_court",
                                        agg_dict={"id": "count", "confidence_scaled": lambda x: list(x),
                                                  "legal_area": lambda x: list(x)})
    conf_df = get_mean_count_legal_area(legal_area_df, "conf_legal_area", get_conf_dict_legal_area)
    for la in LEGAL_AREAS:
        conf_df[la] = conf_df[la].apply(np.mean)
    distribution_df = get_mean_count_legal_area(legal_area_df, "count_legal_area", get_count_dict_legal_area)
    distribution_df[LEGAL_AREAS] = distribution_df[LEGAL_AREAS].div(legal_area_df["id"], axis=0)
    return distribution_df[["lower_court"] + LEGAL_AREAS], conf_df[["lower_court"] + LEGAL_AREAS]


def get_mean_count_legal_area(dataset: pd.DataFrame, col: str, func):
    """
    Applies either get_conf_dict_legal_area or get_count_dict_legal_area to rows.
    Returns Dataframe.
    """
    dataset[col] = dataset.apply(lambda row: func(
        {row["confidence_scaled"][i]: row["legal_area"][i] for i in range(len(row["confidence_scaled"]))},), axis=1)
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


def create_lc_la_distribution_plot(lang: str, lc_la_df: pd.DataFrame):
    """
    Creates lower_court legal_area distribution plot.
    """
    plots.mean_plot_2(lang, len(lc_la_df.columns),
                      lc_la_df, cols=LEGAL_AREAS,
                      ax_labels=["Lower Court", ""],
                      x_labels=lc_la_df.columns,
                      legend_texts=tuple([la.replace("_", " ") for la in LEGAL_AREAS]),
                      title="Distribution of Lower Courts in Legal Areas",
                      filepath=f'plots/lc_distribution_la_{lang}.png')


def create_ttest_plot(lang: str, mu_df: pd.DataFrame, sample_df: pd.DataFrame, filepath: str, col, title):
    """
    @todo
    """
    plots.ttest_plot(lang, scores.ttest(group_to_list(sample_df, "lower_court", col), mu_df, col), "pvalue",
                     "lower_court", 0.05,
                     label_texts=["P-value", "Lower Courts"],
                     title=title,
                     filepath=filepath)


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


def group_to_list(dataset: pd.DataFrame, col_1, col_2) -> pd.DataFrame:
    return dataset.groupby(col_1)[col_2].apply(list).reset_index()


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


def split_oppose_support(dataset_1: pd.DataFrame, dataset_2: pd.DataFrame):
    false_classification_pos, false_classification_neg = dataset_2[
                                                             dataset_2["confidence_direction"] == 1], \
                                                         dataset_2[
                                                             dataset_2["confidence_direction"] == -1]
    o_judgement, s_judgement = dataset_1[dataset_1["numeric_label"] == 1], dataset_1[
        dataset_1["numeric_label"] == -1]
    o_judgement, s_judgement = o_judgement[["id", "explainability_label", "numeric_label", "occluded_text"]], \
                               s_judgement[["id", "explainability_label", "numeric_label", "occluded_text"]]
    o_judgement, s_judgement = pd.merge(false_classification_pos, o_judgement, on="id",
                                        suffixes=(f'_model', f'_human'),
                                        how="inner").drop_duplicates(), \
                               pd.merge(false_classification_neg, s_judgement, on="id",
                                        suffixes=(f'_model', f'_human'),
                                        how="inner").drop_duplicates()
    return o_judgement, s_judgement


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
    @Todo
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
        return np.NaN


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
        preprocessing.write_json(Path(f"{l}/quantitative/annotation_analysis_{l}.json"), counts)

        # Prepare csv tables
        label_mean_df, pers_mean_df = get_label_mean_df(l), get_pers_mean_df(l)
        facts_count_df = count_mean_fact_length(df_gold)
        legal_area_count_df = get_legal_area_distribution(df_gold)
        preprocessing.write_csv_from_list(Path(f"{l}/quantitative/annotation_analysis_{l}.csv"),
                                          [label_mean_df, pers_mean_df,
                                           facts_count_df, legal_area_count_df])
        # Prepare plot tables
        pers_mean_df = pers_mean_df.drop(["label", "mean_token"], axis=1).fillna(0).groupby("annotator").sum()
        plots.mean_plot_1(facts_count_df.drop(f"mean_legal_area").drop(f"mean_year", axis=1),
                          labels=["Years", "Number of Tokens"],
                          legend=[la.replace("_", " ") for la in LEGAL_AREAS],
                          title="Token over legal area and year Distribution ",
                          mean_lines={},
                          filepath=f"plots/mean_tokens_year_legal_area_{l}.png")

        legend = [f"mean {label.lower()}" for label in LABELS_OCCLUSION[:-1]]
        plots.mean_plot_1(pers_mean_df,
                          labels=["Annotators", "Number of Tokens"],
                          legend=legend + [f"mean tokens {label.lower()}"
                                           for label in LABELS_OCCLUSION[:-1]],
                          title="Token Distribution of Annotation Labels",
                          mean_lines={f"mean_tokens_{label.lower().replace(' ', '_')}":
                                          label_mean_df[label_mean_df["label"] == label]["mean_token"].item()
                                      for label in LABELS_OCCLUSION[:-1]},
                          filepath=f"plots/mean_tokens_exp_labels_{l}.png")

        multilingual_mean_df_list.append(label_mean_df.reset_index())
    multilingual_annotation_analysis(multilingual_mean_df_list)


def multilingual_annotation_analysis(df_list: list):
    """
    Prepares language Dataframes for plots.
    Creates multilingual plots.
    """
    df = df_list[0].merge(df_list[1], on="index", how="inner", suffixes=(f'_{LANGUAGES[0]}', f'_{LANGUAGES[1]}'))
    df = df.merge(df_list[2], on="index", how="inner").rename(columns={'mean_token': f'mean_token_{LANGUAGES[2]}'})
    df.drop([f"label_{LANGUAGES[0]}", f"label_{LANGUAGES[1]}", "index"], axis=1, inplace=False)
    plots.mean_plot_2("", len(LABELS_OCCLUSION), df.set_index("label").T,
                      cols=[f"mean_token_{lang}" for lang in LANGUAGES],
                      ax_labels=["Explainability Labels", "Number of Tokens"],
                      x_labels=LABELS_OCCLUSION,
                      legend_texts=tuple((f"Mean Number of Tokens {i}",) for i in LANGUAGES),
                      title="Token Distribution of Annotation Labels in Gold Standard Dataset.",
                      filepath=f"plots/mean_tokens_exp_labels_gold.png")


def lower_court_analysis():
    """
    This functions prepares and performs the quantitative lower_court analysis.
    Dumps individual core numbers into a json.
    Creates plots for quantitative analysis.
    """
    for l in LANGUAGES:
        lower_court_df = preprocessing.read_csv(OCCLUSION_PATHS["analysis"][0].format(l, l), "index")
        baseline_df = preprocessing.get_baseline(lower_court_df)
        lower_court_df = preprocessing.remove_baseline(lower_court_df)

        # Prepare json scores
        lower_court_dict = {"number_of_lower_courts": count_lower_court(baseline_df)[0],
                            "mean_tokens_lower_court": count_lower_court(baseline_df)[1],
                            "number_of_has_flipped_lc(T/F)": list(count_has_flipped(lower_court_df))}
        preprocessing.write_json(Path(f"{l}/quantitative/lower_court_analysis_{l}.json"), lower_court_dict)

        # Lower court distribution plots
        distribution_df = get_lower_court_distribution(lower_court_df)
        mean_pos_df = get_one_sided_agg(lower_court_df[lower_court_df["confidence_direction"] > 0], distribution_df)
        mean_neg_df = get_one_sided_agg(lower_court_df[lower_court_df["confidence_direction"] < 0], distribution_df)

        plots.distribution_plot_1(l, distribution_df, col_x="lower_court", col_y="count",
                                  label_texts=["Lower Court", "Occurrence of Lower Courts in Dataset"],
                                  title=f"Lower Court distribution in the {l} Dataset",
                                  filepath=f'plots/lc_distribution_{l}.png')
        plots.lower_court_effect_plot(mean_pos_df, mean_neg_df, col_y="confidence_direction",
                                      col_x="lower_court",
                                      label_texts=["Confidence Direction", "Lower Court"],
                                      legend_texts=["Positive Influence", "Negative Influence"],
                                      title=f"Effect of Lower Courts on the Prediction Confidence",
                                      filepath=f'plots/lower_court_effect_{l}.png')
        plots.lower_court_effect_plot(mean_pos_df, mean_neg_df, col_y="mean_norm_explainability_score",
                                      col_x="lower_court",
                                      label_texts=["Mean explainability score for one direction", "Lower Court"],
                                      legend_texts=["Exp_Score > 0", "Exp_Score < 0"],
                                      title=f"Mean Distribution on Explainability Scores in both directions",
                                      filepath=f'plots/lc_mean_explainability_score_{l}.png')

        # Lower court legal area distribution plots
        legal_area_distribution, legal_area_conf = get_lc_la_distribution(
            lower_court_df.drop(["norm_explainability_score", "confidence_direction"], axis=1))

        create_lc_la_distribution_plot(l, legal_area_distribution.set_index("lower_court").T)
        create_lc_la_distribution_plot(l, legal_area_conf.set_index("lower_court").T)

        # T-test plots
        baseline_df_mean = group_by_agg_column(baseline_df, "lower_court",
                                               agg_dict={"id": "count", "confidence_scaled": "mean"})
        create_ttest_plot(l, baseline_df_mean, lower_court_df, filepath=f'plots/lc_ttest_conf_score_{l}.png',
                          col="confidence_scaled", title="Distribution of p-values over lower courts.")

        baseline_df_mean["norm_explainability_score"] = 0
        create_ttest_plot(l, baseline_df_mean, lower_court_df, filepath=f'plots/lc_ttest_exp_score_{l}.png',
                          col="norm_explainability_score", title="Distribution of p-values over lower courts.")


def occlusion_analysis():
    """
    @Todo Comment & clean up
    """
    for l in LANGUAGES:
        occlusion_dict = {}
        for nr in NUMBER_OF_EXP:
            occlusion_df = preprocessing.read_csv(OCCLUSION_PATHS["analysis"][1].format(l, nr, l), "index")

            baseline_df = occlusion_df[occlusion_df["explainability_label"] == "Baseline"]
            occlusion_df = preprocessing.remove_baseline(occlusion_df)
            occlusion_df = occlusion_df[occlusion_df["confidence_direction"] != 0]  # ignore neutral classification
            occlusion_df, false_classification, score_1, score_0 = preprocessing.get_correct_direction(occlusion_df)
            # Prepare json scores
            occlusion_dict = {"number of correct classification": score_1,
                              "number of incorrect_classification": score_0}
            preprocessing.write_json(Path(f"{l}/quantitative/occlusion_analysis_{l}.json"), occlusion_dict)

            preprocessing.write_csv_from_list(Path(f"{l}/quantitative/occlusion_analysis_{l}.csv"),
                                              [count_occlusion_tokens(l)])

            # @todo Plot of correct and incorrect classifications

            o_judgement, s_judgement = split_oppose_support(occlusion_df, false_classification)
            o_judgement, s_judgement = scores.calculate_IAA_occlusion(o_judgement, l), \
                                       scores.calculate_IAA_occlusion(s_judgement, l)
