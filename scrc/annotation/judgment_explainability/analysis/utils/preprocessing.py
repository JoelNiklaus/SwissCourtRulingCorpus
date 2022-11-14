import ast
import itertools
import json
from pathlib import Path
from random import randint

import numpy as np
import pandas
import pandas as pd
from netcal.scaling import TemperatureScaling

# Constant variable definitions
LANGUAGES = ["de", "fr", "it"]
PERSONS = ["angela", "lynn", "thomas"]
SESSIONS = ["angela", "lynn", "thomas", "gold_nina"]
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
AGGREGATIONS = ["mean", "max", "min"]
NAN_KEY = 10000

pd.options.mode.chained_assignment = None  # default='warn'


def extract_dataset(filepath_a, filepath_b) -> dict:
    """
    Tries to extracts data from JSONL file to a dictionary list.
    Transforms dictionary list to dataframe.
    Catches file not found.
    Returns a dictionary of dataframes with filename as keys.
    """
    datasets = {}
    for language in LANGUAGES:
        try:
            json_list = read_jsnol(filepath_a.format(language, language))
            # List of dict to dataframe
            dfItem = pd.DataFrame.from_records(json_list)
            dfItem = dfItem.set_index("id_scrc")
            dfItem.index.name = f"annotations_{language}"
            datasets[f"annotations_{language}"] = dfItem
        except FileNotFoundError:
            pass
        for session in SESSIONS:
            try:
                json_list = read_jsnol(filepath_b.format(language, language, session))
                dfItem = pd.DataFrame.from_records(json_list)
                dfItem.index.name = f"annotations_{language}-{session}"
                datasets[f"annotations_{language}-{session}"] = dfItem
            except FileNotFoundError:
                pass
    return datasets


def read_jsnol(filename: str) -> list:
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
    Writes a JSONL file from dictionary list.
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
    Writes csv from Dataframe.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=True, index_label="index")


def write_json(filepath: Path, dictionary: dict):
    """
    Writes a json from dict.
    """
    json_object = json.dumps(dictionary, indent=4)
    with open(filepath, "w") as outfile:
        outfile.write(json_object)


def get_tokens_dict(df: pandas.DataFrame, col_1: str, col_2: str, new_col: str) -> pandas.DataFrame:
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


def extract_values_from_column(annotations: pandas.DataFrame, col_1: str, col_2: str) -> pandas.DataFrame:
    """
    Extracts values from list of dictionaries in columns (by explode), resets index and drops col_2.
    Extracts dictionaries from row (apply) and adds corresponding prefix.
    Drops original column (col_1).
    Joins full dataframe with new column and returns it.
    """
    annotations_col = annotations.explode(col_1).reset_index().drop([col_2], axis=1)
    df_col = annotations_col[col_1].apply(pd.Series).add_prefix(f"{col_1}_")
    annotations_col = annotations_col.drop([col_1], axis=1)
    return annotations_col.join(df_col)


def get_span_df(annotations_spans: pandas.DataFrame, annotations_tokens: pandas.DataFrame, span: str,
                lang: str) -> pandas.DataFrame:
    """
    Extract all rows where span_label matches the span given as parameter (e.g. span = "Lower court").
    Queries list of values from chosen rows and creates list of token numbers (token ids) of span start and end number.
    Adds token ids to dict.
    Extracts token ids from dict and gets corresponding word tokens for the ids.
    Appends word tokens and ids as Dataframe to list.
    Returns Dataframe containing ids and words of spans.
    """
    spans = annotations_spans[annotations_spans["spans_label"] == span]
    token_numbers = {}
    for mini_list in list(
            spans[[f'annotations_{lang}', '_annotator_id', 'spans_token_start', 'spans_token_end']].values):
        numbers = []
        # Range of numbers between spans_token_start spans_token_end
        for nr in list(range(int(mini_list[2]), int(mini_list[3]) + 1)):
            numbers.append(nr)
        token_numbers[f"{mini_list[0]}.{mini_list[1]}.{randint(0, 100000)}"] = numbers
    spans_list = []
    for key in token_numbers:
        new_annotations_tokens = annotations_tokens[
            annotations_tokens[f'annotations_{lang}'] == int(key.split(".")[0])].copy()
        new_annotations_tokens = new_annotations_tokens[new_annotations_tokens["tokens_id"].isin(token_numbers[key])]
        new_annotations_tokens = new_annotations_tokens[new_annotations_tokens['_annotator_id'] == key.split(".")[1]]
        spans_list.append(new_annotations_tokens)
    return pd.concat(spans_list), token_numbers


def group_columns(df: pandas.DataFrame, lang: str) -> pandas.DataFrame:
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


def get_white_space_dicts(ws: pd.DataFrame, index: str) -> pd.DataFrame:
    """
    Creates new column 'id_ws_dict'. @Todo
    """
    ws = get_tokens_dict(ws, 'tokens_id', 'tokens_ws', 'id_ws_dict')[[index, 'id_ws_dict']]
    ws['tokens_ws_dict'] = ws.groupby([index])['id_ws_dict'].transform(
        lambda x: "{{{}}}".format(','.join(x.astype(str)).replace("{", "").replace("}", "")))
    return ws.drop('id_ws_dict', axis=1).drop_duplicates()


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


def get_annotator_df(annotator_name: str, tokens: pd.DataFrame, lang: str, version: str) -> pd.DataFrame:
    """
    Copies entries from Dataframe from specific annotator.
    Groups tokens_text, tokens_id, tokens_dict from getone case together.
    Creates Dataframe containing ids, 'tokens_text','tokens_id', 'tokens_dict'.
    Drops duplicates
    Transforms tokens_id string to list.
    Returns Dataframe
    """
    if version == "1":
        annotator = tokens[
            tokens['_annotator_id'] == f"annotations_{lang}-{annotator_name}"].drop_duplicates().copy()
    if version == "2":
        annotator = tokens[
            tokens['_annotator_id'] == f"annotations_{lang}_inspect-{annotator_name}"].drop_duplicates().copy()
    if version == "3":
        annotator = tokens[
            tokens['_annotator_id'] == f"annotations_{lang}_inspect-{annotator_name}"].drop_duplicates().copy()
        annotator = annotator.append(tokens[
                                         tokens[
                                             '_annotator_id'] == f"annotations_{lang}-{annotator_name}"].drop_duplicates().copy())

    annotator = group_columns(annotator, lang)
    annotator = annotator[[f'annotations_{lang}', 'tokens_text', 'tokens_id', 'tokens_dict', 'tokens_ws_dict']]
    annotator = annotator.drop_duplicates()
    annotator["tokens_id"] = annotator["tokens_id"].astype(str).str.split(",")
    no_duplicates = []
    for lst in annotator["tokens_id"].values:
        lst = list(dict.fromkeys(lst))
        no_duplicates.append(lst)
    annotator["tokens_id"] = no_duplicates
    return annotator


def remove_duplicate(list_duplicate: list) -> list:
    """
    Removes duplicates from list, returns list
    """
    return list(dict.fromkeys(list_duplicate))


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
    Copies Dataframe and joins annotators tokens_dict.
    Creates a token dictionary from the joined individual token dictionaries where each word token has a different value.
    Adds normalized tokens dictionaries to dataframe.
    Returns Dataframe.
    """
    normalized_tokens = []
    df_copy = df.copy()
    for token_dicts in df_copy[
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
    max_length = find_max_length(list_of_list)[1]
    for lst in list_of_list:
        index = list(list_of_list).index(lst)
        if NAN_KEY not in lst:
            while len(lst) < max_length:
                lst.append(token_dict["Nan"])
            list_of_list[index] = lst

    return list_of_list


def find_max_length(lst: list) -> (list, int):
    """
    Returns max list from list of list and max list length from list oft lists.
    """
    return max(x for x in lst), max(len(x) for x in lst)


def temp_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces the judgment labels with int 0,1.
    Creates two NumPy 1-D and 2-D arrays.
    Applies temprature scaling to the values and returns calibrated DataFra,e

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
