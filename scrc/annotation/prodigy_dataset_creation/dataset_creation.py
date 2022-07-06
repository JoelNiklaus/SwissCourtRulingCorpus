import re
import sys

import pandas as pd
import psycopg2
# Load environment variable from .env
from dotenv import load_dotenv

load_dotenv()

from dataset_creation_functions import LANGUAGES, \
    read_JSONL, write_JSONL, db_stream, filter_dataset, filter_new_cases, \
    set_id_scrc

FILEPATH_ANNOTATION = "../judgment_explainability/annotations/annotations_{}.jsonl"
FILEPATH_DATASET_JE = "../judgment_explainability/datasets/annotation_input_set_{}.jsonl"
FILEPATH_DATASET_P = "../judgment_prediction/datasets/prediction_input_set_{}.jsonl"

if __name__ == '__main__':
    usage = "Arguments:\n - language: the language you want to extract the dataset from, e.g. the relation de. " \
            "\n - mode 0 for judment explainability, 1 for judgment outcome" \
            "Type -all to create dataset for all the possible languages. \n " \
            "Type -new to generate a new dataset which replaces ignored and rejected cases. \n" \
            "Usage:\n python datasetCreation.py language mode\n" \
            "Example:\n python datasetCreation.py de 0"
    try:
        assert len(sys.argv) == 3
        ids_scrc = []
        try:
            if sys.argv[1] == "-all":
                for l in LANGUAGES:
                    filepath_dataset = FILEPATH_DATASET_JE.format(l)
                    FILEPATH_ANNOTATION = FILEPATH_ANNOTATION.format(l)
                    if int(sys.argv[2]) == 1:
                        ids_scrc = set_id_scrc(pd.DataFrame.from_records(read_JSONL(FILEPATH_ANNOTATION)), 1)
                        filepath_dataset= FILEPATH_DATASET_P.format(l)
                    dataset = filter_dataset(db_stream(l, int(sys.argv[2]), ids_scrc), int(sys.argv[2]))
                    write_JSONL(filepath_dataset.format(l), dataset)
            if sys.argv[1] == "-new":
                for l in LANGUAGES:
                    filepath_dataset = FILEPATH_DATASET_JE.format(l)
                    FILEPATH_ANNOTATION = FILEPATH_ANNOTATION.format(l)
                    ids_scrc = set_id_scrc(pd.DataFrame.from_records(read_JSONL(FILEPATH_ANNOTATION)), 0)
                    dataset = db_stream(l, int(sys.argv[2]), ids_scrc)
                    new_case_list = filter_new_cases(pd.DataFrame.from_records(read_JSONL(FILEPATH_ANNOTATION)),
                                                     dataset)
                    if int(sys.argv[2]) == 1:
                        filepath_dataset= FILEPATH_DATASET_P.format(l)
                    write_JSONL(filepath_dataset.format(l), new_case_list+read_JSONL(FILEPATH_DATASET_JE.format(l)))
            if sys.argv[1] in LANGUAGES:
                FILEPATH_ANNOTATION = FILEPATH_ANNOTATION.format(sys.argv[1])
                if int(sys.argv[2]) == 1:
                    filepath_dataset = FILEPATH_DATASET_P.format(sys.argv[1])
                    ids_scrc = set_id_scrc(pd.DataFrame.from_records(read_JSONL(FILEPATH_ANNOTATION)), 1)
                filepath_dataset = FILEPATH_DATASET_JE.format(sys.argv[1])
                dataset = filter_dataset(db_stream(sys.argv[1], int(sys.argv[2]), ids_scrc), int(sys.argv[2]))

                write_JSONL(filepath_dataset.format(sys.argv[1]), dataset)

        except psycopg2.errors.UndefinedTable as err:
            print("Invalid language argument:\n {0}".format(err))
            print(usage)
    except AssertionError:
        print(usage)
