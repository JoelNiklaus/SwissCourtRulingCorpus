import os
import sys
import json

import pandas
import pandas as pd
import psycopg2
from psycopg2 import sql

'''
sys.path.append('../../')
from scrc.utils.main_utils import get_legal_area
sys.path.pop()
'''
# Load environment variable from .env
from dotenv import load_dotenv

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

LEGAL_AREAS = ["penal_law", "social_law", "civil_law"]
# One ruling per 3 legal areas 2 outcomes and 6 years
NUMBER_OF_RULINGS_PER_LANGUAGE = 3 * 2 * 6

PATH_TO_PREDICTIONS = r"sjp/finetune/xlm-roberta-base-hierarchical/de,fr,it,en/"
PATH_TO_DATASET_SCRC = r"dataset_scrc/"

PREDICTIONS_TEST_PATHS =["3/de/predictions_test.csv", "3/fr/predictions_test.csv", "3/it/predictions_test.csv"] # Random seed 3 has highest macro F1-score


LANGUAGES = ["de", "fr", "it"]

# Writes a JSONL file from dictionary list
def write_JSONL(filename: str, data: list):
    with open(filename, 'w') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')
    print("Successfully saved file " + filename)

def read_csv(filepath:str) -> pandas.DataFrame:
    df = pd.read_csv(filepath)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.set_index("id")
    return df

def join_dataframes(df_1: pandas.DataFrame, df_2: pandas.DataFrame, lang:str) -> pandas.DataFrame:
    df = df_1.join(df_2, lsuffix='_caller', rsuffix='_other')
    if lang != LANGUAGES[2]:
        df = df.drop(df[df["is_correct"] == False].index)
    df = df[df.columns.drop(list(df.filter(regex='_other')))]
    df = df.drop(["origin_canton", "origin_court","origin_chamber","num_tokens_spacy","num_tokens_bert","chamber","origin_region" ], axis=1)
    df = df[df["legal_area"].isin(LEGAL_AREAS)]
    df["text"] = df["text"].str.replace('\"\"', '\"')
    df = df.dropna()

    return df

def validate_filtering(tuple_list: list,filtered_list: list):
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
        assert len(filtered_list) == NUMBER_OF_RULINGS_PER_LANGUAGE
    except AssertionError:
        print("Dataset was not filtered correctly: Contains an incorrect number of entries {}.".format(len(filtered_list)))
        for t in tuple_list:
            for y in incomplete_set_years:
                if y == t[0]:
                    print(t)

# Filters dictionary list using unique tuple as condition
# Asserts a list length smaller than 36
# Asserts uniqueness of dataset entries
# Returns filtered list
def filter_dataset(data: list) -> list:
    tuples = set()
    filtered_dataset = []
    for d in data:
        t = ( d["year"],d["legal_area"], d["judgment"])
        if t not in tuples:
            tuples.add(t)
            filtered_dataset.append(d)

    validate_filtering(sorted(tuples),filtered_dataset)
    return filtered_dataset


# Connects to scrc database, executes sql query and uses batched fetching to returns results.
# Query results are ordered by size of the facts (shortest facts first).
# Preprocesses results by adding the corresponding legal area and filtering out double judgments.
# Returns filtered records from the database as list.
def db_stream(language: str) -> list:
    unfiltered_dataset = []
    prediction_test = read_csv(PATH_TO_PREDICTIONS + PREDICTIONS_TEST_PATHS[LANGUAGES.index(language)])
    test_set = read_csv(PATH_TO_DATASET_SCRC+language+"/test.csv")
    val_set = read_csv(PATH_TO_DATASET_SCRC+language+"/val.csv")
    test_val_set = pd.concat([join_dataframes(prediction_test, test_set, language),join_dataframes(PREDICTIONS_EVAL, val_set,language)])

    # scrc database connection configuration string
    config = "dbname=scrc user={} password={} host=localhost port=5432".format(DB_USER, DB_PASSWORD)

    # prodigy database connection configuration string, to fetch processed ids

    query = sql.SQL("SELECT id, date, html_url, pdf_url, facts, considerations, rulings " \
                    "FROM {} WHERE (date between '2015-01-01' AND '2020-12-31') " \
                    "AND (CHAR_LENGTH(facts)>500)" \
                    "AND NOT (CHAR_LENGTH(rulings)=0 )" \
                    "AND NOT (CHAR_LENGTH(rulings)=0)"\
                    "ORDER BY CHAR_LENGTH(facts)").format(sql.Identifier(language))

    with psycopg2.connect(config) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            while True:
                rows = cursor.fetchmany(1000)
                if not rows:
                    break
                for row in rows:
                    idx, date, html_url, pdf_url, facts, considerations, rulings= row
                    if str(html_url) != "":
                        link = html_url
                    if str(pdf_url) != "":
                        link = pdf_url

                    res = {"id_scrc": idx,
                           "text": facts,
                           "year": int(date.year),
                           "link": str(link),
                           "considerations": considerations,
                           "ruling": rulings}

                    if res["text"] in set(test_val_set["text"]):
                        res["legal_area"] = list(test_val_set.loc[test_val_set["text"] == res["text"]]["legal_area"])[0]
                        res["judgment"] = list(test_val_set.loc[test_val_set["text"] == res["text"]]["label_caller"])[0]
                        res["confidence"] = list(test_val_set.loc[test_val_set["text"] == res["text"]]["confidence"])[0]
                        res["id_csv"] = list(test_val_set.loc[test_val_set["text"] == res["text"]].index)[0]
                        res["is_correct"] = list(test_val_set.loc[test_val_set["text"] == res["text"]]["is_correct"])[0]

                        unfiltered_dataset.append(res)

            print(len(unfiltered_dataset))
            unfiltered_dataset = sorted(unfiltered_dataset, key=lambda x: x["confidence"], reverse=True)
            return filter_dataset(unfiltered_dataset)


if __name__ == '__main__':
    usage = "Arguments:\n - language: the language you want to extract the dataset from, e.g. the relation de. " \
            "Type -all to create dataset for all the possible languages. \n " \
            "Usage:\n python datasetCreation.py language\n" \
            "Example:\n python datasetCreation.py de"
    PREDICTIONS_EVAL = read_csv(PATH_TO_PREDICTIONS + "2/predictions_eval.csv")
    try:
        assert len(sys.argv) == 2
        try:
            if sys.argv[1] == "-all":
                for l in LANGUAGES:
                    dataset = db_stream(l)
                    write_JSONL('annotation_input_set_' + l + '.jsonl', dataset)
            else:
                dataset = db_stream(sys.argv[1])
                write_JSONL('annotation_input_set_' + sys.argv[1] + '.jsonl', dataset)

        except psycopg2.errors.UndefinedTable as err:
            print("Invalid language argument:\n {0}".format(err))
            print(usage)
    except AssertionError:
        print(usage)
