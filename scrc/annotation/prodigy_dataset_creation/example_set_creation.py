import os
import sys
import json
import re
import pandas
import pandas as pd
import psycopg2
from psycopg2 import sql
import numpy
# Load environment variable from .env
from dotenv import load_dotenv
load_dotenv()

from dataset_creation_functions import CONFIG, header_preprocessing, PATTERNS, read_csv, PATH_TO_PREDICTIONS, write_JSONL
from dataset_creation import PATH_TO_JE_DATASET

def db_stream_example() -> list:
    example_list = []
    query = sql.SQL("SELECT id, date, file_number, html_url, pdf_url, header, facts, judgments " \
                    "FROM de WHERE (date < '2015-01-01') " \
                    "AND (CHAR_LENGTH(facts)>2000)")

    with psycopg2.connect(CONFIG) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            while True:
                rows = cursor.fetchmany(1000)
                if not rows:
                    break
                for row in rows:
                    idx, date, file_number, html_url, pdf_url, header, facts, judgments = row
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
                           "header": header_preprocessing(header, PATTERNS["de"]),
                           "judgment": judgments}
                    example_list.append(res)
            return example_list[100:]

if __name__ == '__main__':
    usage = "Arguments:\n - language: the language you want to extract the dataset from, e.g. the relation de. " \
            "Type -all to create dataset for all the possible languages. \n " \
            "Type -ex to create an example dataset from the training set. \n"\
            "Type -new to generate a new dataset which replaces ignored and rejected cases. \n"\
            "Usage:\n python datasetCreation.py language\n" \
            "Example:\n python datasetCreation.py de"
    PREDICTIONS_EVAL = read_csv(PATH_TO_PREDICTIONS + "2/predictions_eval.csv")
    try:
        assert len(sys.argv) == 2
        try:
            if sys.argv[1] == "-ex":
                dataset= db_stream_example()
                write_JSONL('{}/annotation_input_set_de_ex.jsonl'.format(PATH_TO_JE_DATASET), dataset)
        except psycopg2.errors.UndefinedTable as err:
            print("Invalid language argument:\n {0}".format(err))
            print(usage)
    except AssertionError:
        print(usage)
