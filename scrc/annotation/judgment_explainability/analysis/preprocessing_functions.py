import itertools
import json
import re
from pathlib import Path
from random import randint
import ast
import pandas
import pandas as pd

# Constant variable definitions
LANGUAGES = ["de", "fr", "it"]
PERSONS = ["angela", "lynn", "thomas"]
SESSIONS = ["angela", "lynn", "thomas", "gold_final", "gold_nina"]
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
            json_list = read_JSONL(filepath_a.format(language, language))
            # List of dict to dataframe
            dfItem = pd.DataFrame.from_records(json_list)
            dfItem = dfItem.set_index("id_scrc")
            dfItem.index.name = "annotations_{}".format(language)
            datasets["annotations_{}".format(language)] = dfItem
        except FileNotFoundError:
            pass
        for session in SESSIONS:
            try:
                json_list = read_JSONL(filepath_b.format(language, language, session))
                dfItem = pd.DataFrame.from_records(json_list)
                dfItem.index.name = "annotations_{}-{}".format(language, session)
                datasets["annotations_{}-{}".format(language, session)] = dfItem
            except FileNotFoundError:
                pass
    return datasets


def read_JSONL(filename: str) -> list:
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


def write_JSONL(filename: str, data: list):
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


def dump_user_input(dataset_dict: dict):
    """
    Dumps all the user inputs as csv.
    Catches key errors.
    """
    for data in dataset_dict:
        try:
            lang = re.search("de|fr|it", str(data)).group(0)
            user_input = dataset_dict[data][dataset_dict[data]["user_input"] != ""]
            write_csv(Path("{}/{}_user_input.csv".format(lang, dataset_dict[data].index.name)),
                      user_input[['id_scrc', '_annotator_id', "user_input"]])
            print("Saved {}.csv successfully!".format(dataset_dict[data].index.name + "_user_input"))
        except KeyError:
            pass


def dump_case_not_accepted(dataset_dict: dict):
    """
    Dumps all the all not accepted cases as csv.
    Catches key errors.
    """
    for data in dataset_dict:
        try:
            lang = re.search("de|fr|it", str(data)).group(0)
            case_not_accepted = dataset_dict[data][dataset_dict[data]["answer"] != "accept"]
            write_csv(Path("{}/{}_ig_re.csv".format(lang, dataset_dict[data].index.name)),
                      case_not_accepted[['id_scrc', '_annotator_id', "user_input"]])
            print("Saved {}.csv successfully!".format(dataset_dict[data].index.name + "_ig_re"))
        except KeyError:
            pass


def write_csv(filepath: Path, df: pd.DataFrame):
    """
    Creates a csv from Dataframe and saves it.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=True,index_label="index")


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
    df_col = annotations_col[col_1].apply(pd.Series).add_prefix("{}_".format(col_1))
    annotations_col = annotations_col.drop([col_1], axis=1)
    return annotations_col.join(df_col)


def get_span_df(annotations_spans: pandas.DataFrame, annotations_tokens: pandas.DataFrame, span: str, lang: str) -> (
pandas.DataFrame, list):
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
            spans[['annotations_{}'.format(lang), '_annotator_id', 'spans_token_start', 'spans_token_end']].values):
        numbers = []
        # Range of numbers between spans_token_start spans_token_end
        for nr in list(range(int(mini_list[2]), int(mini_list[3]) + 1)):
            numbers.append(nr)
        token_numbers["{}.{}.{}".format(mini_list[0], mini_list[1], randint(0, 100000))] = numbers
    spans_list = []
    for key in token_numbers:
        new_annotations_tokens = annotations_tokens[
            annotations_tokens['annotations_{}'.format(lang)] == int(key.split(".")[0])].copy()
        new_annotations_tokens = new_annotations_tokens[new_annotations_tokens["tokens_id"].isin(token_numbers[key])]
        new_annotations_tokens = new_annotations_tokens[new_annotations_tokens['_annotator_id'] == key.split(".")[1]]
        spans_list.append(new_annotations_tokens)
    spans = pd.concat(spans_list)
    return spans, token_numbers


def group_columns(df: pandas.DataFrame, lang: str) -> pandas.DataFrame:
    """
    Groups columns tokens_text, tokens_id and tokens_dict by same index (e.g. annotations_de).
    Each column is joint differently (e.g. tokens_dict joint as string dictionary).
    Returns Dataframe.
    """
    df['tokens_text'] = df.groupby(['annotations_{}'.format(lang)])['tokens_text'].transform(
        lambda x: ' '.join(x))
    df['tokens_id'] = df.groupby(['annotations_{}'.format(lang)])['tokens_id'].transform(
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
    @Todo
    """
    ws = get_tokens_dict(ws, 'tokens_id', 'tokens_ws', 'id_ws_dict')[[index, 'id_ws_dict']]
    ws['tokens_ws_dict'] = ws.groupby([index])['id_ws_dict'].transform(lambda x: "{{{}}}".format(','.join(x.astype(str)).replace("{", "").replace("}", "")))
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