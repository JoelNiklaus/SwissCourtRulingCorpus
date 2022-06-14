import sys
import re
import pandas as pd
import psycopg2

# Load environment variable from .env
from dotenv import load_dotenv
load_dotenv()


from dataset_creation_functions import LANGUAGES,\
    read_JSONL, write_JSONL, db_stream, filter_dataset, filter_new_cases,\
    set_id_scrc
PATH_TO_JE_DATASET = "../judgment_explainability/datasets"
PATH_TO_JP_DATASET = "../judgment_prediction/datasets"

if __name__ == '__main__':
    usage = "Arguments:\n - language: the language you want to extract the dataset from, e.g. the relation de. " \
            "\n - mode 0 for judment explainability, 1 for judgment outcome"\
            "Type -all to create dataset for all the possible languages. \n " \
            "Type -new to generate a new dataset which replaces ignored and rejected cases. \n"\
            "Usage:\n python datasetCreation.py language mode\n" \
            "Example:\n python datasetCreation.py de 0"
    try:
        assert len(sys.argv) == 3
        if int(sys.argv[2]) == 0:
            filename =  'annotation_input_set_{}.jsonl'
        if int(sys.argv[2]) == 1:
             filename = 'prediction_input_set_{}.jsonl'
        try:
            if sys.argv[1] == "-all":
                for l in LANGUAGES:
                    dataset = filter_dataset(db_stream(l,int(sys.argv[2]),[]),int(sys.argv[2]))
                    write_JSONL(filename.format(l), dataset)

            if sys.argv[1] == "-new":
                filepath = "../annotations/annotations_de.jsonl"
                ids_scrc = set_id_scrc(pd.DataFrame.from_records(read_JSONL(filepath)))
                l = re.findall(r"(?<=_)(.*)(?=.jsonl)",filepath)[0]
                dataset = db_stream(l,int(sys.argv[2]), ids_scrc)
                new_case_list = filter_new_cases(pd.DataFrame.from_records(read_JSONL(filepath)), dataset)
                write_JSONL(filename.format(l), new_case_list)
            else:
                dataset = filter_dataset(db_stream(sys.argv[1],int(sys.argv[2]),[]),int(sys.argv[2]))
                write_JSONL(filename.format(sys.argv[1]), dataset)

        except psycopg2.errors.UndefinedTable as err:
            print("Invalid language argument:\n {0}".format(err))
            print(usage)
    except AssertionError:
        print(usage)
