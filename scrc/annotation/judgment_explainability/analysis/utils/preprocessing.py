import ast
import itertools
import json
import math
import warnings
from decimal import Decimal
from pathlib import Path
from random import randint

import numpy as np
import pandas as pd
from netcal.scaling import TemperatureScaling
from scipy.stats import stats

warnings.filterwarnings("ignore")

"""
This is a collection of helper functions used by the diffrent components of the analysis, 
experiment_creator and prodigy_dataset_creator.
"""

LANGUAGES = ["de", "fr", "it"]
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
PERSONS = ["angela", "lynn", "thomas"]
SESSIONS = ["angela", "lynn", "thomas", "gold_nina"]
NAN_KEY = 10000


def extract_dataset(filepath_a: str, filepath_b: str) -> dict:
    """
    Extracts data from JSONL file and transforms it to a Dataframe.
    Excepts file not found.
    Returns a dictionary of dataframes with filenames as keys.
    """
    datasets = {}
    for language in LANGUAGES:
        try:
            json_list = read_jsonl(filepath_a.format(language, language))
            # List of dict to dataframe
            dfItem = pd.DataFrame.from_records(json_list)
            dfItem = dfItem.set_index("id_scrc")
            dfItem.index.name = f"annotations_{language}"
            datasets[f"annotations_{language}"] = dfItem
        except FileNotFoundError:
            pass
        for session in SESSIONS:
            try:
                json_list = read_jsonl(filepath_b.format(language, language, session))
                dfItem = pd.DataFrame.from_records(json_list)
                dfItem.index.name = f"annotations_{language}-{session}"
                datasets[f"annotations_{language}-{session}"] = dfItem
            except FileNotFoundError:
                pass
    return datasets


def extract_prediction_test_set(prediction_path: str, test_set_path: str):
    """
    Gets testset and prediction from csv.
    Returns merged testset and prediction set.
    """
    prediction = temp_scaling(read_csv(prediction_path, "id")).reset_index().reset_index().drop(["label"], axis=1)
    test_set = read_csv(test_set_path, "id")
    return pd.merge(prediction, test_set, on="index",
                    suffixes=(f'_test_set', f'_prediction'),
                    how="outer").drop_duplicates()


def read_jsonl(filename: str) -> list:
    """
    Reads JSONL file and returns a dataframe
    """
    with open(filename, "r") as json_file:
        json_list = list(json_file)
    a_list = []
    for json_str in json_list:
        result = json.loads(json_str)
        a_list.append(result)
        assert isinstance(result, dict)

    return a_list


def write_jsonl(filename: str, data: list):
    """
    Writes a JSONL file from list of dictionaries.
    """
    with open(filename, 'w') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')
    print("Successfully saved file " + filename)


def read_csv(filepath: str, index: str) -> pd.DataFrame:
    """
    Reads CSV file sets index and returns a DataFrame.
    """
    df = pd.read_csv(filepath)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.set_index(index)
    return df


def write_csv(filepath: Path, df: pd.DataFrame):
    """
    Writes CSV file from Dataframe.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=True, index_label="index")


def write_csv_from_list(path: Path, df_list: list):
    """
    Writes csv file from Dataframe list.
    """
    with open(path, "w") as f:
        f.truncate()
        for df in df_list:
            df.to_csv(f)
            f.write("\n")


def write_json(filepath: Path, dictionary: dict):
    """
    Writes a json from dict.
    """
    json_object = json.dumps(dictionary, indent=4)
    with open(filepath, "w") as outfile:
        outfile.write(json_object)


def get_accepted_cases(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Returns accepted cases from dataset.
    """
    return dataset[dataset["answer"] == "accept"]


def remove_baseline(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Returns Dataframe w/o baseline entries.
    """
    return dataset[dataset["explainability_label"] != "Baseline"]


def get_baseline(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Returns baseline Dataframe.
    """
    return dataset[dataset["explainability_label"] == "Baseline"]


def group_to_list(dataset: pd.DataFrame, col_1, col_2) -> pd.DataFrame:
    return dataset.groupby(col_1)[col_2].apply(list).reset_index()


def group_by_agg_column(dataset: pd.DataFrame, col_1, agg_dict: dict) -> pd.DataFrame:
    """
    Groups by col_1 and applies count of "id" and mean of 'confidence_scaled'.
    Returns Dataframe
    """
    return dataset.groupby(col_1).agg(agg_dict).reset_index()


def annotation_preprocessing(filepaths: list, lang: str, dataset_name: str) -> (
        pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Extract datasets from jsonl, gets acceoted cases.
    Separates spans and tokens columns and joins them into dictionary.
    Returns extracted dataset, spans Dataframe and tokens Dataframe.
    """
    datasets = extract_dataset(filepaths[0], filepaths[1])
    dataset = get_accepted_cases(datasets[dataset_name])
    dataset.index.name = f"annotations_{lang}"
    return dataset, extract_values_from_column(dataset, "spans", "tokens"), extract_values_from_column(dataset,
                                                                                                       "tokens",
                                                                                                       "spans")


def extract_values_from_column(df: pd.DataFrame, col_1: str, col_2: str) -> pd.DataFrame:
    """
    Extracts values from list of dictionaries in columns (by explode), resets index and drops col_2.
    Extracts dictionaries from row (apply) and adds corresponding prefix.
    Drops original column (col_1).
    Joins full dataframe with new column and returns it.
    """
    exploded_cols = df.explode(col_1).reset_index().drop([col_2], axis=1)
    df_col = exploded_cols[col_1].apply(pd.Series).add_prefix(f"{col_1}_")
    exploded_cols = exploded_cols.drop([col_1], axis=1)
    return exploded_cols.join(df_col)


def get_span_df(spans_df: pd.DataFrame, tokens_df: pd.DataFrame, span: str,
                lang: str) -> (pd.DataFrame, dict):
    """
    Extract all rows where span_label matches the span given as parameter (e.g. span = "Lower court").
    Queries list of values from chosen rows and creates list of token numbers (token ids) of span start and end number.
    Extracts token ids from dict key and gets corresponding word tokens for the ids.
    Returns Dataframe containing ids and words of spans and token number dict.
    """
    spans = spans_df[spans_df["spans_label"] == span]
    token_numbers = {}
    for mini_list in list(
            spans[[f'annotations_{lang}', '_annotator_id', 'spans_token_start', 'spans_token_end']].values):
        numbers = []
        # Range of numbers between spans_token_start spans_token_end
        for nr in list(range(int(mini_list[2]), int(mini_list[3]) + 1)):
            numbers.append(nr)
        token_numbers[
            f"{mini_list[0]}.{mini_list[1]}.{randint(0, 100000)}"] = numbers  # Adds token ids to dict with randomized key
    spans_list = []
    for key in token_numbers:
        new_annotations_tokens = tokens_df[
            tokens_df[f'annotations_{lang}'] == int(key.split(".")[0])].copy()
        new_annotations_tokens = new_annotations_tokens[new_annotations_tokens["tokens_id"].isin(token_numbers[key])]
        new_annotations_tokens = new_annotations_tokens[new_annotations_tokens['_annotator_id'] == key.split(".")[1]]
        spans_list.append(new_annotations_tokens)
    return pd.concat(spans_list), token_numbers


def join_to_dict(df: pd.DataFrame, col_1: str, col_2: str, new_col: str) -> pd.DataFrame:
    """
    Joins two columns of a dataframe to dictionary string (str({row[col_1]: row[col_2]})).
    Appends strings to list.
    Returns Dataframe with new column from list.
    """
    dict_list = []
    for index, row in df.iterrows():
        token_dict = str({row[col_1]: row[col_2]})
        dict_list.append(token_dict)

    df[new_col] = dict_list
    return df


def get_label_df(lang: str, spans_df: pd.DataFrame, tokens_df: pd.DataFrame, label: str):
    label_df = get_span_df(spans_df, tokens_df, label, lang)[0]
    # Getting the necessary dictionaries
    ws_df = get_white_space_dicts(label_df, "annotations_{}".format(lang))
    label_df = join_to_dict(label_df, "tokens_id", "tokens_text", "tokens_dict")
    label_df = label_df.join(ws_df[[f"annotations_{lang}", 'tokens_ws_dict']].set_index(f"annotations_{lang}"),
                             on="annotations_{}".format(lang))
    return label_df


def get_white_space_dicts(ws_df: pd.DataFrame, index: str) -> pd.DataFrame:
    """
    Creates new column 'id_ws_dict' by joining columns 'tokens_id' and 'tokens_ws'
    using @join_to_dict.
    Returns Dataframe containing new column.
    """
    ws_df = join_to_dict(ws_df, 'tokens_id', 'tokens_ws', 'id_ws_dict')[[index, 'id_ws_dict']]
    ws_df.loc[:, 'tokens_ws_dict'] = ws_df.groupby([index])['id_ws_dict'].transform(
        lambda x: "{{{}}}".format(','.join(x.astype(str)).replace("{", "").replace("}", "")))
    return ws_df.drop('id_ws_dict', axis=1).drop_duplicates()


def group_token_columns(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """
    Groups columns tokens_text, tokens_id and tokens_dict by same index (e.g. annotations_de).
    Each column is joint differently (e.g. tokens_dict joint as string dictionary).
    Returns Dataframe.
    """
    df['tokens_text'] = df.groupby([f'annotations_{lang}'])['tokens_text'].transform(
        lambda x: ' '.join(x))
    df['tokens_id'] = df.groupby([f'annotations_{lang}'])['tokens_id'].transform(
        lambda x: ','.join(x.astype(str)))
    df['tokens_dict'] = df.groupby([f'annotations_{lang}'])['tokens_dict'].transform(
        lambda x: "{{{}}}".format(','.join(x.astype(str)).replace("{", "").replace("}", "")))
    return df


def apply_functions_lower_court(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies normalize_string to ccluded_text.
    Gets original court for each row and inserts lower_court as occlusion
    Returns Dataframe.
    """
    df["occluded_text"] = df.apply(
        lambda row: normalize_string(row["occluded_text"]), axis=1)
    df["lower_court"] = df.apply(
        lambda row: get_original_court(row["lower_court"], row["occluded_text"]), axis=1)
    df["facts"] = df.apply(
        lambda row: row["facts"].replace("[*]", row["lower_court"] + " "), axis=1)
    return df


def get_original_court(lower_court, original_court):
    """
    Returns the "original_court", if the lower_court is in the occluded text".
    Returns lower_court if not.
    """
    if lower_court == original_court:
        return "original court"
    else:
        return lower_court


def string_to_dict(df: pd.DataFrame, col_name) -> pd.DataFrame:
    """
    Transforms column of string dictionary to column of dictionary.
    Returns Dataframe.
    """
    dict_list = []
    for token_dict in df[col_name].values:
        if type(token_dict) == str and token_dict != "Nan":
            token_dict = ast.literal_eval(token_dict)
        dict_list.append(token_dict)
    df[col_name] = dict_list
    return df


def get_combinations(val_list: list, length_subset: int) -> list:
    """
    Gets combinations of a list of values and returns them.
    """
    combinations = []
    for L in range(0, len(val_list) + 1):
        for subset in itertools.combinations(val_list, L):
            if len(subset) == length_subset and [NAN_KEY] not in subset and ["Nan"] not in subset:
                combinations.append(subset)

    return combinations


def get_annotator_df(tokens_df: pd.DataFrame, lang: str, annotator: str, version: str) -> pd.DataFrame:
    """
    Copies entries from Dataframe from specific annotator.
    Groups tokens_text, tokens_id, tokens_dict from getone case together.
    Creates Dataframe containing ids, 'tokens_text','tokens_id', 'tokens_dict'.
    Drops duplicates
    Transforms tokens_id string to list.
    Returns Dataframe
    """
    annotator_df = tokens_df
    version_ids = {"1": "annotations_{}-{}", "2": "annotations_{}_inspect-{}"}
    if version in ["1", "2"]:
        annotator_df = tokens_df[
            tokens_df['_annotator_id'] == version_ids[version].format(lang, annotator)].drop_duplicates().copy()
    if version == "3":
        annotator_df = tokens_df[
            tokens_df['_annotator_id'] == version_ids["1"].format(lang, annotator)].drop_duplicates().copy()
        annotator_df = annotator_df \
            .append(tokens_df[tokens_df['_annotator_id'] == version_ids["2"].format(lang, annotator)]
                    .drop_duplicates().copy())

    annotator_df = group_token_columns(annotator_df, lang)
    annotator_df = annotator_df[[f'annotations_{lang}', 'tokens_text', 'tokens_id', 'tokens_dict', 'tokens_ws_dict']]
    annotator_df = annotator_df.drop_duplicates()
    annotator_df["tokens_id"] = annotator_df["tokens_id"].astype(str).str.split(",")
    no_duplicates = []
    for lst in annotator_df["tokens_id"].values:
        lst = list(dict.fromkeys(lst))
        no_duplicates.append(lst)
    annotator_df["tokens_id"] = no_duplicates
    return annotator_df


def merge_triple(df_list: list, person_suffixes: list, lang: str):
    """
    Merges first and second Dataframe using outer join.
    Formats column names using person_suffixes, fills Nan values with "Nan".
    Repeats the same merge with the new merged Dataframe and the third Dataframe.
    Returns merged Dataframe.
    """
    i = 0
    merged_df = pd.merge(df_list[i], df_list[i + 1], on=f"annotations_{lang}",
                         suffixes=(f'_{person_suffixes[i]}', f'_{person_suffixes[i + 1]}'),
                         how="outer").fillna("Nan")

    return pd.merge(merged_df, df_list[i + 2], on=f"annotations_{lang}", how="outer").fillna("Nan").rename(
        columns={"tokens_text": f"tokens_text_{person_suffixes[i + 2]}",
                 "tokens_id": f"tokens_id_{person_suffixes[i + 2]}",
                 "tokens_dict": f"tokens_dict_{person_suffixes[i + 2]}",
                 'tokens_ws_dict': f'tokens_ws_dict_{person_suffixes[i + 2]}'})


def get_normalize_tokens_dict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins tokens_dicts of each annotator.
    Creates a token dictionary from the joined individual token dictionaries where each token has a different value.
    Returns Dataframe containing column of normalized tokens dictionaries.
    """
    normalized_tokens = []
    for token_dicts in df.copy()[
        [f"tokens_dict_{PERSONS[0]}", f"tokens_dict_{PERSONS[1]}", f"tokens_dict_{PERSONS[2]}"]].values:
        tokens = []
        for token_dict in token_dicts:
            if token_dict != "Nan":
                for token in eval(token_dict).values():
                    tokens.append(token)
            if token_dict == "Nan":
                tokens.append(token_dict)
        normalized_tokens.append(dict(zip(tokens, range(0, len(tokens)))))
    df["normalized_tokens_dict"] = normalized_tokens
    return df


def normalize_person_tokens(df: pd.DataFrame, pers: str, lang: str) -> pd.DataFrame:
    """
    Extracts tokens_text for given person and tokenizes list.
    Gets token dictionary of all persons from Dataframe and adds "Nan" value.
    Asserts that there are no duplicated values in dict after adding "Nan".
    Gets id for each token in persons token list and appends it to normalized_tokens_row.
    Appends normalized_tokens_row to normalized_tokens which creates new column in Dataframe.
    Drops duplicates and returns Dataframe.
    """
    normalized_tokens = []
    for sentences in df[[f"annotations_{lang}", f"tokens_dict_{pers}"]].values:
        normalized_tokens_row = []
        tokens = [sentences[1]]
        if sentences[1] != "Nan":
            tokens = eval(sentences[1]).values()
        token_dict = df[df[f"annotations_{lang}"] == sentences[0]]["normalized_tokens_dict"].values[0]
        token_dict["Nan"] = NAN_KEY
        assert len(token_dict.values()) == len(set(token_dict.values()))
        for word in tokens:
            normalized_tokens_row.append(token_dict[word])
        normalized_tokens.append(normalized_tokens_row)

    df[f"normalized_tokens_{pers}"] = normalized_tokens
    df = df.loc[df.astype(str).drop_duplicates().index]
    return df


def normalize_list_length(list_of_list: list, token_dict: dict) -> (list, list):
    """
    Appends "Nan" to normalize list length (make them same length).
    Returns lists.
    """
    max_length = max(len(x) for x in list_of_list)
    for lst in list_of_list:
        index = list(list_of_list).index(lst)
        if NAN_KEY in lst:
            while len(lst) < max_length:
                lst.append(token_dict["Nan"])
            list_of_list[index] = lst

    return list_of_list


def temp_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces the judgment labels with int 0,1.
    Creates two NumPy 1-D and 2-D arrays.
    Applies temperature scaling to the values and returns calibrated DataFra,e

    Uses TemperatureScaling() from Kueppers et al.
    via https://github.com/EFS-OpenSource/calibration-framework#calibration-framework
    """
    df["prediction"] = np.where(df["prediction"] == "dismissal", 0, 1)
    ground_truth = np.array(df["prediction"].values)  # ground truth digits between 0-1 - shape: (n_samples,)
    confidences = np.array(
        df[["prediction", "confidence"]].values)  # confidence estimates between 0-1 - shape: (n_samples, n_classes)

    temperature = TemperatureScaling()
    temperature.fit(confidences, ground_truth)
    df["confidence_scaled"] = temperature.transform(confidences)
    return df


def occlusion_preprocessing(lang: str, df: pd.DataFrame, filename: str):
    """
    @todo
    """
    df = df.set_index("index")
    df = calculate_explainability_score(df)
    df = find_flipped_cases(df)
    df_0, df_1 = df[df["prediction"] == 0].sort_values(by=['explainability_score'],
                                                       ascending=True), \
                 df[df["prediction"] == 1].sort_values(by=['explainability_score'],

                                                       ascending=False)
    df_0, df_1 = get_confidence_direction(df_0, 0), get_confidence_direction(
        df_1, 1)
    df_0, df_1 = get_norm_explainability_score(df_0, 0), get_norm_explainability_score(df_1, 1)

    write_csv(Path(f"{lang}/occlusion/{filename}.csv"), df_0.append(df_1))


def get_correct_direction(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
    """
    Adds numeric_label column with values (-1: Supports judgement, 0: Neutral, 1: Opposes judgement).
    Adds correct_direction boolean column (True if numeric_label == confidence direction).
    Splits df into direction sets (0: False, 1: True).
    Returns Dataframe, direction set 0, grouped set 1 and grouped set 0.
    """
    df["numeric_label"] = np.where(df["explainability_label"] == LABELS[1], -1, df["explainability_label"])
    df["numeric_label"] = np.where(df["explainability_label"] == LABELS[2], 1, df["numeric_label"])
    df["numeric_label"] = np.where(df["explainability_label"] == "Neutral", 0, df["numeric_label"])
    df["correct_direction"] = np.where(df["numeric_label"] == df["confidence_direction"], True, False)
    s_0, s_1 = df[df["correct_direction"] == False], df[df["correct_direction"] == True]
    s_1 = s_1.groupby("explainability_label")["correct_direction"].count()
    return df, s_0, s_1, s_0.groupby("explainability_label")["correct_direction"].count()


def get_confidence_direction(df: pd.DataFrame, prediction: int):
    """
    Returns Dataframe with column confidence_direction (-1, 0, 1).
    """
    df["confidence_direction"] = df["explainability_score"].apply(
        lambda row: normalize_exp_score_direction(row, prediction)[0])
    return df


def get_norm_explainability_score(df: pd.DataFrame, prediction: int):
    """
    Returns Dataframe with column norm_explainability_score.
    """
    df["norm_explainability_score"] = df["explainability_score"].apply(
        lambda row: normalize_exp_score_direction(row, prediction)[1])
    return df


def normalize_exp_score_direction(explainability_score: float, prediction: int) -> (int, float):
    """
    Calculates direction of confidence depending on prediction and explainability score (e.g. for prediction 0
    explainability_score <0 means less confidence).
    Normalizes explainability_score (flips sign of explainability_score for prediction 1)
    Returns confidence direction and normalized explainability_score.
    Meaning of return values:
    1: More confident than baseline
    0: equally as confident as baseline
    -1: less confident than baseline
    """
    if explainability_score == 0 or math.isnan(explainability_score):
        return 0, explainability_score
    if prediction == 0:
        if explainability_score < 0:
            return -1, explainability_score
        if explainability_score > 0:
            return 1, explainability_score
    else:
        if explainability_score < 0:
            return 1, abs(explainability_score)
        if explainability_score > 0:
            return -1, (-1) * explainability_score


def calculate_explainability_score(df: pd.DataFrame):
    """
    Separates baseline entries and non baseline entries.
    Calculates difference between the baseline confidence of a case and the confidence of the occlusion.
    Adds explainability_score as column to Dataframe, appends baseline back to it and returns it.
    """
    score_list = []
    baseline_df = get_baseline(df)
    occlusion_df = remove_baseline(df)
    for index, row in occlusion_df.iterrows():
        baseline_value = baseline_df[baseline_df["id"] == row["id"]]["confidence"].max()
        score_list.append(float(baseline_value) - float(row["confidence"]))  # diffrence between baseline and occlusion
    occlusion_df.loc[:, "explainability_score"] = score_list
    occlusion_df = occlusion_df.append(baseline_df)
    return occlusion_df


def get_one_sided_agg(dataset: pd.DataFrame, sorted_df: pd.DataFrame, col):
    """
    Gets mean of one sided explainability_score and confidence_direction.
    Returns Dataframe.
    """
    dataset = group_by_agg_column(dataset, col,
                                  agg_dict={"confidence_scaled": "mean", "confidence_direction": "mean",
                                            "norm_explainability_score": "mean"}) \
        .rename(columns={"norm_explainability_score": "mean_norm_explainability_score"})
    if col == "lower_court":
        dataset = sorted_df.merge(dataset, on="lower_court", how="inner")
    dataset["confidence_direction"] = dataset["confidence_direction"] * dataset["confidence_scaled"]
    return dataset


def find_flipped_cases(df: pd.DataFrame):
    """
    Separates baseline entries and non baseline entries.
    Adds boolean column has_flipped (True when baseline prediction equals occlusion prediction).
    Appends baseline back to Dataframe and returns it.
    """
    score_list = []
    baseline_df = get_baseline(df)
    occlusion_df = remove_baseline(df)
    for index, row in occlusion_df.iterrows():
        baseline_value = baseline_df[baseline_df["id"] == row["id"]]["prediction"].max()
        if row["prediction"] == baseline_value:
            score_list.append(False)
        else:
            score_list.append(True)
    occlusion_df.loc[:, "has_flipped"] = score_list
    occlusion_df = occlusion_df.append(baseline_df)
    return occlusion_df


def group_by_flipped(dataset: pd.DataFrame, col) -> pd.DataFrame:
    """
    Groups by col.
    Returns Dataframe with proportional count of flipped and unflipped rows.
    """
    has_flipped_df = group_by_agg_column(dataset[dataset["has_flipped"] == True], col, {"has_flipped": "count"})
    dataset = group_by_agg_column(dataset, col, {"id": "count"})
    dataset = dataset.merge(has_flipped_df, on=col)
    dataset["has_not_flipped"] = (dataset["id"] - dataset["has_flipped"])
    dataset["has_flipped"] = dataset["has_flipped"]
    return dataset


def normalize_string(string: str) -> str:
    """
    Returns string w/o trailing whitespace.
    """
    if string[-1] == ' ':
        if string[-2] == ' ':
            return string[:-2]
        else:
            return string[:-1]
    if string[-1] != ' ':
        return string


def ttest(sample_df: pd.DataFrame, mu_df, col):
    """
    @todo
    """
    tc_list = []
    pvalue_list = []
    for lower_court in sample_df.values:
        mu = mu_df[mu_df["lower_court"] == lower_court[0]][col].values[0]
        result = stats.ttest_1samp(lower_court[1], popmean=mu)
        tc_list.append(result.statistic)
        pvalue_list.append(result.pvalue)
    sample_df["tc"] = tc_list
    sample_df["pvalue"] = pvalue_list
    sample_df["pvalue"] = sample_df["pvalue"].apply(Decimal)
    return sample_df
