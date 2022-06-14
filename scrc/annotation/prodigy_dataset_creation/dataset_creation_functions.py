import json
import os
import re

import numpy
import pandas
import pandas as pd
import psycopg2
# Load environment variable from .env
from dotenv import load_dotenv
from psycopg2 import sql

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# scrc database connection configuration string
CONFIG = "dbname=scrc user={} password={} host=localhost port=5432".format(DB_USER, DB_PASSWORD)

LEGAL_AREAS = ["penal_law", "social_law", "civil_law"]
LANGUAGES = ["de", "fr", "it"]

# One ruling per 3 legal areas 2 outcomes and 6 years
NUMBER_OF_RULINGS_PER_LANGUAGE_explainability = 3 * 2 * 6
# Two ruling (model correct and incorrect) per 3 legal areas 2 outcomes and 6 years
NUMBER_OF_RULINGS_PER_LANGUAGE_prediction = 2 * 3 * 2 * 6

PATH_TO_PREDICTIONS = "sjp/finetune/xlm-roberta-base-hierarchical/de,fr,it,en/"
PATH_TO_DATASET_SCRC = "dataset_scrc/"
PREDICTIONS_TEST_PATHS = ["3/de/predictions_test.csv", "3/fr/predictions_test.csv",
                          "3/it/predictions_test.csv"]  # Random seed 3 has highest macro F1-score

# Patterns to add formatting to the facts
PATTERNS = {"de": "[u|U]rteil \w+ \d+?.? \w+ \d{4}",
            "fr": "([a|A]rrêt \w+ \d+?.? \w+ \d{4})|([a|A]rrêt \w+ \d+?er \w+ \d{4})",
            "it": "([s|S]entenza \w+ \d+?.? \w+ \d{4})|([s|S]entenza \w+'\d+?.? \w+ \d{4})"}


# Writes a JSONL file from dictionary list.
def write_JSONL(filename: str, data: list):
    with open(filename, 'w') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')
    print("Successfully saved file " + filename)


# Reads JSONL file and returns a dataframe
def read_JSONL(filename: str) -> list:
    with open(filename, "r") as json_file:
        json_list = list(json_file)
    a_list = []
    for json_str in json_list:
        result = json.loads(json_str)
        a_list.append(result)
        assert isinstance(result, dict)

    return a_list


# Reads CSV file sets index to "id" and returns a DataFrame.
def read_csv(filepath: str) -> pandas.DataFrame:
    df = pd.read_csv(filepath)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.set_index("id")
    return df


def set_id_scrc(df: pd.DataFrame) -> numpy.ndarray:
    df = df[df["answer"] != "accept"]
    return df["id_scrc"].values


# Joins to dataframes according to their ids.
# Drops rows of False prediction (except in italian dataset).
# Drops duplicated, redundant and Nan columns.
# Checks legal area.
# returns joined DataFrame.
def join_dataframes(df_1: pandas.DataFrame, df_2: pandas.DataFrame, lang: str, mode: int) -> pandas.DataFrame:
    df = df_1.join(df_2, lsuffix='_caller', rsuffix='_other')
    df = df.sort_values("is_correct", ascending=False)
    if lang == LANGUAGES[0] and mode == 0:
        df = df.drop(df[df["is_correct"] == False].index)
    df = df[df.columns.drop(list(df.filter(regex='_other')))]
    df = df.drop(["origin_canton", "origin_court", "origin_chamber", "num_tokens_spacy", "num_tokens_bert", "chamber",
                  "origin_region"], axis=1)
    df = df[df["legal_area"].isin(LEGAL_AREAS)]
    df["text"] = df["text"].str.replace('\"\"', '\"')
    df = df.dropna()
    return df


def filter_dataset(data: list, mode: int) -> list:
    """
    Filters dictionary list using unique tuple as condition.
    Returns filtered dictionary list.
    """
    tuples = set()
    filtered_dataset = []
    print(mode)
    for d in data:
        if mode == 0:
            t = (d["year"], d["legal_area"], d["judgment"])
        if mode == 1:
            t = (d["year"], d["legal_area"], d["judgment"], d["is_correct"])
            print(t)
        if t not in tuples:
            tuples.add(t)
            filtered_dataset.append(d)

    validate_filtering(sorted(tuples), filtered_dataset)
    return filtered_dataset

def validate_filtering(tuple_list: list, filtered_list: list):
    """
    Asserts uniqueness of dataset entries
    Checks if every year has a complete set of 6 rulings
    Asserts a list length smaller than NUMBER_OF_RULINGS_PER_LANGUAGE
    """

    i = 0
    incomplete_set_years = []
    try:
        assert len(tuple_list) == len(set(tuple_list))
    except AssertionError:
        print("Dataset was not filtered correctly: Contains duplicates.")
    try:
        while i < len(tuple_list):
            if tuple_list[i][0] == tuple_list[i + 5][0]:
                print("Set {} complete!".format(tuple_list[i][0]))
                i += 6
            else:
                i += 1
                if tuple_list[i][0] != tuple_list[i + 1][0]:
                    print("Set {} not complete!".format(tuple_list[i][0]))
                    incomplete_set_years.append(tuple_list[i][0])
        assert len(filtered_list) == NUMBER_OF_RULINGS_PER_LANGUAGE_explainability
    except AssertionError:
        print("Dataset was not filtered correctly: Contains an incorrect number of entries {}.".format(
            len(filtered_list)))
        for t in tuple_list:
            for y in incomplete_set_years:
                if y == t[0]:
                    print(t)

def filter_new_cases(df: pd.DataFrame, data: list) -> list:
    tuples = set()
    filtered_dataset = []
    df = df[df["answer"] != "accept"]
    values = df[["year", "legal_area", "judgment"]].values
    for value in values:
        tuples.add(tuple(value))
    for d in data:
        t = (d["year"], d["legal_area"], d["judgment"])
        if t in tuples:
            filtered_dataset.append(d)
            tuples.remove(t)
    return filtered_dataset


def header_preprocessing(h: str, pattern: str) -> str:
    result = re.search(pattern, h)
    if result is not None:
        return result.group(0)
    else:
        return h


def add_sections(section: str) -> str:
    return re.sub("(\ )(?=[B-Z]{1}\. {1})", "\n", section)



def db_stream(language: str, mode: int, ids_scrc) -> list:
    """
    Connects to scrc database, executes sql query and uses batched fetching to returns results.
    Query results are ordered by size of the facts (shortest facts first).
    Matches results with entries from test_val_set
    Preprocesses matched results by adding the corresponding legal area and judgment.
    Returns filtered records from the database as list.
    """
    unfiltered_dataset = []
    predictions_eval = read_csv(PATH_TO_PREDICTIONS + "2/predictions_eval.csv")
    prediction_test = read_csv(PATH_TO_PREDICTIONS + PREDICTIONS_TEST_PATHS[LANGUAGES.index(language)])
    test_set = read_csv(PATH_TO_DATASET_SCRC + language + "/test.csv")
    val_set = read_csv(PATH_TO_DATASET_SCRC + language + "/val.csv")
    test_val_set = pd.concat(
        [join_dataframes(prediction_test, test_set, language, mode),
         join_dataframes(predictions_eval, val_set, language, mode)])

    query = sql.SQL("SELECT id, date, file_number, html_url, pdf_url, header, facts " \
                    "FROM {} WHERE (date between '2015-01-01' AND '2020-12-31') " \
                    "AND (CHAR_LENGTH(facts)>2000)" \
                    "ORDER BY CHAR_LENGTH(facts)").format(sql.Identifier(language))

    with psycopg2.connect(CONFIG) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            while True:
                rows = cursor.fetchmany(1000)
                if not rows:
                    break
                for row in rows:
                    idx, date, file_number, html_url, pdf_url, header, facts = row
                    if idx not in ids_scrc:
                        if str(html_url) != "":
                            link = html_url
                        if str(pdf_url) != "":
                            link = pdf_url

                        res = {"id_scrc": idx,
                               "text": facts,
                               "facts_length": len(facts),
                               "year": int(date.year),
                               "file_number": file_number,
                               "link": str(link),
                               "header": header_preprocessing(header, PATTERNS[language])}
                        if res["text"] in set(test_val_set["text"]):
                            res["legal_area"] = \
                            list(test_val_set.loc[test_val_set["text"] == res["text"]]["legal_area"])[0]
                            res["judgment"] = \
                            list(test_val_set.loc[test_val_set["text"] == res["text"]]["label_caller"])[0]
                            res["id_csv"] = list(test_val_set.loc[test_val_set["text"] == res["text"]].index)[0]
                            res["is_correct"] = \
                            list(test_val_set.loc[test_val_set["text"] == res["text"]]["is_correct"])[0]

                            res["text"] = add_sections(res["text"])
                            if res not in unfiltered_dataset:
                                unfiltered_dataset.append(res)

            print(len(unfiltered_dataset))
            return unfiltered_dataset
