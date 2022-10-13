
import os
import re
import sys
from collections import Counter
import numpy
import pandas
import pandas as pd
import psycopg2
import ast
# Load environment variable from .env
from dotenv import load_dotenv
load_dotenv()
from psycopg2 import sql

from scrc.annotation.judgment_explainability.analysis.preprocessing_functions import read_csv, read_JSONL, write_JSONL


FILEPATH_ANNOTATION = "../judgment_explainability/legal_expert_annotations/{}/annotations_{}.jsonl"
FILEPATH_DATASET_JE = "../judgment_explainability/annotation_datasets/annotation_input_set_{}.jsonl"
FILEPATH_DATASET_P = "../judgment_prediction/annotation_datasets/prediction_input_set_{}.jsonl"

PATH_TO_PREDICTIONS = os.getenv("PATH_TO_PREDICTIONS")
PATH_TO_DATASET_SCRC = os.getenv("PATH_TO_DATASET_SCRC")
PREDICTIONS_TEST_PATHS = ast.literal_eval(os.getenv("PREDICTIONS_TEST_PATHS"))
PREDICTION_EVAL = read_csv(PATH_TO_PREDICTIONS + "2/predictions_eval.csv", "id")

# scrc database connection configuration string
CONFIG = f'dbname=scrc user={os.getenv("DB_USER")} password={os.getenv("DB_PASSWORD")} host=localhost port=5432'

LEGAL_AREAS = ["penal_law", "social_law", "civil_law"]

# Patterns to add formatting to the facts
PATTERNS = {"de": "[u|U]rteil \w+ \d+?.? \w+ \d{4}",
            "fr": "([a|A]rrêt \w+ \d+?.? \w+ \d{4})|([a|A]rrêt \w+ \d+?er \w+ \d{4})",
            "it": "([s|S]entenza \w+ \d+?.? \w+ \d{4})|([s|S]entenza \w+'\d+?.? \w+ \d{4})"}

LANGUAGES =  ast.literal_eval(os.getenv("LANGUAGES"))

IDS_SCRC = []

def set_id_scrc(df: pd.DataFrame) -> numpy.ndarray:
    """
    Returns a list of all used ids
    """
    df = df[df["answer"] != "accept"]
    return df["id_scrc"].values


def join_dataframes(df_1: pandas.DataFrame, df_2: pandas.DataFrame, lang: str, mode: str) -> pandas.DataFrame:
    """
    Joins to dataframes according to their ids.
    Drops rows of False prediction in german.
    Drops duplicated, redundant and Nan columns.
    Checks legal area and returns jpined DataFrame.
    """
    df = df_1.join(df_2, lsuffix='_caller', rsuffix='_other')
    # Use only correct cases in german judgement explainability
    if lang == "de" and mode == "je":
        df = df.drop(df[df["is_correct"] == False].index)
    df = df[df.columns.drop(list(df.filter(regex='_other')))]
    df = df.drop(["origin_canton", "origin_court", "origin_chamber", "num_tokens_spacy", "num_tokens_bert", "chamber",
                  "origin_region"], axis=1)
    df = df[df["legal_area"].isin(LEGAL_AREAS)]
    df["text"] = df["text"].str.replace('\"\"', '\"')
    return df.dropna()


def filter_dataset(data: list, mode: str) -> list:
    """
    Filters dictionary list using unique tuple as condition.
    Returns filtered and validated dictionary list.
    """
    tuples = set()
    filtered_dataset = []
    # Adds only unique tuples to set and appends corresponding data to filtered list
    for d in data:
        t = (d["year"], d["legal_area"], d["judgment"])
        if mode == "jp":
            t = (d["year"], d["legal_area"], d["judgment"], str(d["is_correct"]))
        if t not in tuples:
            tuples.add(t)
            filtered_dataset.append(d)

    validate_filtering(data, sorted(tuples), filtered_dataset, mode)
    return filtered_dataset


def validate_filtering(original_data: list, tuple_list: list, filtered_list: list, mode: str):
    """
    Asserts uniqueness of dataset entries.
    Checks if every year has a complete set of 6 or 12 rulings.
    If a year is incomplete fill_incomplete_sets() is triggered.
    Asserts a list length smaller than NUMBER_OF_RULINGS_PER_LANGUAGE.
    """
    # For mode je there are six ruling per year for mode jp 12
    i, j, steps = 0, 5,6

    if mode == "jp":
        j, steps = 11, 12
    incomplete_set_years = []
    assert len(tuple_list) == len(set(tuple_list))
    try:
        while i < len(tuple_list):
            # Checks cases per year
            if tuple_list[i][0] == tuple_list[i + j][0]:
                print("Set {} complete!".format(tuple_list[i][0]))
                i += steps
            else:
                i += 1
                if tuple_list[i][0] != tuple_list[i + 1][0]:
                    print("Set {} not complete!".format(tuple_list[i][0]))
                    incomplete_set_years.append(tuple_list[i][0])

        if len(incomplete_set_years) > 0:
            fill_incomplete_sets(original_data, filtered_list, tuple_list, mode)

        # One ruling per 3 legal areas 2 outcomes and 6 years
        if mode == "je":
            assert len(filtered_list) == 36
        # Two ruling (model correct and incorrect) per 3 legal areas 2 outcomes and 6 years
        if mode == "jp":
            assert len(filtered_list) == 72

    except AssertionError:
        print("Dataset was not filtered correctly: Contains an incorrect number of entries {}.".format(
            len(filtered_list)))
        for t in tuple_list:
            for y in incomplete_set_years:
                print(incomplete_set_years)
                if y == t[0]:
                    print(t)


def filter_new_cases(df: pd.DataFrame, data: list) -> list:
    """
    Gets the not accepted cases from the dataset.
    Replaces cases with new cases having the same properties.
    Returns filtered dataset with new cases.
    """
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


def fill_incomplete_sets(original_data: list, filtered_dataset: list, current_tuples: list, mode: str):
    """
    Fills incomplete year dataset by first adding cases regardless of 'if_correct' parameter.
    If there are not enough cases (12) to pass validation, more cases are added by disregarding the judgement.
    Validates dataset.
    """
    # Iteration 0: Fills years with different legal_area and judgment but with duplicated is_correct.
    iteration = 0
    if iteration == 0:
        for d in original_data:

            t = (d["year"], d["legal_area"], d["judgment"], "None")
            if t not in current_tuples and d not in filtered_dataset:
                t_1, t_2 = (d["year"], d["legal_area"], d["judgment"], "False"), (
                    d["year"], d["legal_area"], d["judgment"], "True")
                if t_1 not in current_tuples or t_2 not in current_tuples:
                    current_tuples.append(t)
                    filtered_dataset.append(d)
        iteration = + 1
    # Iteration 1: Fills years with different legal_area but with duplicated judgment and is_correct.
    if iteration == 1:
        for d in original_data:
            t = (d["year"], d["legal_area"], "None", "None")
            if t not in current_tuples and d not in filtered_dataset:
                t_1, t_2 = (d["year"], d["legal_area"], "approval", "False"), (
                    d["year"], d["legal_area"], "approval", "True")
                t_3, t_4 = (d["year"], d["legal_area"], "dismissal", "False"), (
                    d["year"], d["legal_area"], "dismissal", "True")
                # Checks if there are not already enough cases (12)
                if t_1 not in current_tuples or t_2 not in current_tuples or t_3 not in current_tuples or t_4 not in current_tuples:
                    if Counter(elem[0] for elem in current_tuples)[t[0]] < 12:
                        current_tuples.append(t)
                        filtered_dataset.append(d)

    validate_filtering(original_data, sorted(current_tuples), filtered_dataset, mode)


def header_preprocessing(h: str, pattern: str) -> str:
    """
    Extracts string from header according to regex pattern
    """
    result = re.search(pattern, h)
    if result is not None:
        return result.group(0)
    else:
        return h


def add_sections(section: str) -> str:
    """
    Adds section according to regex pattern.
    """
    return re.sub("((\ )(?=A\.[a-z]{1}\. {1})|(\ )(?=[B-Z]{1}\. {1}))", "\n", section)


def get_test_val_set(lang: str, mode:str):
    """
    Returns jpined test and validation set from csv.
    """
    return pd.concat(
        [join_dataframes(read_csv(PATH_TO_PREDICTIONS + PREDICTIONS_TEST_PATHS[LANGUAGES.index(lang)], "id"),
                         read_csv(PATH_TO_DATASET_SCRC + lang + "/test.csv", "id"), lang, mode),
         join_dataframes(PREDICTION_EVAL, read_csv(PATH_TO_DATASET_SCRC + lang + "/val.csv", "id"), lang, mode)])


def db_stream(lang: str, mode: str, ids_scrc) -> list:
    """
    Connects to scrc database, executes sql query and uses batched fetching to returns results.
    Query results are ordered by size of the facts (shortest facts first).
    Matches results with entries from TEST_VAL_SET
    Preprocesses matched results by adding the corresponding legal area and judgment.
    Returns filtered records from the database as list.
    """
    unfiltered_dataset = []
    test_val_set = get_test_val_set(lang, mode)


    query = sql.SQL("SELECT id, date, file_number, html_url, pdf_url, header, facts " \
                    "FROM {} WHERE (date between '2015-01-01' AND '2020-12-31') " \
                    "AND (CHAR_LENGTH(facts)>2000)" \
                    "ORDER BY CHAR_LENGTH(facts)").format(sql.Identifier(lang))

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
                               "header": header_preprocessing(header, PATTERNS[lang])}
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

            return unfiltered_dataset



if __name__ == '__main__':
    usage = "Arguments:\n - language: the language you want to extract the dataset from, e.g. de. " \
            "\n - mode je for judment explainability, jp for judgment prediction" \
            "Type -all to create dataset for all the possible languages. \n " \
            "Type -new to generate a new dataset which replaces ignored and rejected cases. \n" \
            "Usage:\n python datasetCreation.py language mode\n" \
            "Example:\n python datasetCreation.py de je" \
            "\n python datasetCreation.py -all je"
    try:
        assert len(sys.argv) == 3
    except AssertionError:
        print(usage)
    language = sys.argv[1]
    judgement_mode = sys.argv[2]
    try:
        if language in LANGUAGES:
            FILEPATH_ANNOTATION = FILEPATH_ANNOTATION.format(language, language)
            filepath_dataset = FILEPATH_DATASET_JE.format(language)
            if judgement_mode == "jp":
                filepath_dataset = FILEPATH_DATASET_P.format(language)
                IDS_SCRC = set_id_scrc(pd.DataFrame.from_records(read_JSONL(FILEPATH_ANNOTATION)))
            dataset = filter_dataset(db_stream(language, judgement_mode, IDS_SCRC), judgement_mode)
            write_JSONL(filepath_dataset.format(language), dataset)
        else:
            for l in LANGUAGES:
                filepath_dataset = FILEPATH_DATASET_JE.format(l)
                FILEPATH_ANNOTATION = FILEPATH_ANNOTATION.format(l,l)
                if judgement_mode == "jp":
                    IDS_SCRC = set_id_scrc(pd.DataFrame.from_records(read_JSONL(FILEPATH_ANNOTATION)))
                    filepath_dataset = FILEPATH_DATASET_P.format(l)
                if language == "-all":
                    # judgement outcome needs new cases, IDS_SCRC is set to all cases in judgment explainability dataset
                    dataset = filter_dataset(db_stream(l, judgement_mode, IDS_SCRC), judgement_mode)
                    write_JSONL(filepath_dataset.format(l), dataset)
                # for new cases IDS_SCRC is set to all the accepted cases ids in the dataset
                if language == "-new":
                    IDS_SCRC = set_id_scrc(pd.DataFrame.from_records(read_JSONL(FILEPATH_ANNOTATION)))
                    dataset = db_stream(l, judgement_mode, IDS_SCRC)
                    new_case_list = filter_new_cases(pd.DataFrame.from_records(read_JSONL(FILEPATH_ANNOTATION)),
                                                     dataset)
                    write_JSONL(filepath_dataset.format(l),
                                new_case_list + read_JSONL(FILEPATH_DATASET_JE.format(l)))
    except psycopg2.errors.UndefinedTable as err:
        print(f"Invalid language argument:\n {err}")
        print(usage)

