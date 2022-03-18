import os
import sys
import json
import psycopg2
from psycopg2 import sql

sys.path.append('../../')
from scrc.utils.main_utils import get_legal_area
sys.path.pop()

# Load environment variable from .env
from dotenv import load_dotenv

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

LEGAL_AREAS = ["penal_law", "social_law", "civil_law"]
# One ruling per 3 legal areas 2 outcomes and 6 years
NUMBER_OF_RULINGS_PER_LANGUAGE = 3 * 2 * 6


# Writes a JSONL file from dictionary list
def write_JSONL(filename: str, data: list):
    with open(filename, 'w') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')
    print("Successfully saved file " + filename)


# Filters dictionary list using unique tuple as condition
# Asserts a list length smaller than 36
# Asserts uniqueness of dataset entries
# Returns filtered list
def filter_dataset(data: list) -> list:
    tuple_list = []
    filtered_list = []
    for d in data:
        t = (d["legal_area"], d["date"], d["judgment"][0])
        if t not in tuple_list:
            tuple_list.append(t)
            filtered_list.append(d)

    tuple_list = sorted(tuple_list)
    '''
    for i in range(0, len(tuple_list)-2,2):
        print(tuple_list[i], tuple_list[i+1])

    print(len(filtered_list))
    '''
    try:
        assert len(tuple_list) == len(set(tuple_list))
    except AssertionError:
        print("Dataset was not filtered correctly: Contains duplicates.")
    try:
        assert len(filtered_list) <= NUMBER_OF_RULINGS_PER_LANGUAGE
    except AssertionError:
        print("Dataset was not filtered correctly: Contains an incorrect number of entries.")

    return filtered_list


# Connects to scrc database, executes sql query and uses batched fetching to returns results.
# Query results are ordered by size of the facts (shortest facts first).
# Preprocesses results by adding the corresponding legal area and filtering out double judgments.
# Returns filtered records from the database as list.
def db_stream(language: str) -> list:
    unfiltered_dataset = []
    # scrc database connection configuration string
    config = "dbname=scrc user={} password={} host=localhost port=5432".format(DB_USER, DB_PASSWORD)

    # prodigy database connection configuration string, to fetch processed ids

    query = sql.SQL("SELECT id, chamber, date, html_url, pdf_url, facts, considerations, rulings, judgements " \
                    "FROM {} WHERE (date between '2015-01-01' AND '2020-12-31') " \
                    "AND NOT (judgements @> '[\"inadmissible\"]' OR judgements @> '[\"partial_approval\"]' " \
                    " OR judgements @> '[\"partial_dismissal\"]'OR judgements@> '[\"write_off\"]'" \
                    "OR judgements @> '[\"unification\"]')" \
                    "AND NOT (judgements @>'null')" \
                    "AND (CHAR_LENGTH(facts)>400)" \
                    "AND NOT (CHAR_LENGTH(rulings)=0 )" \
                    "AND NOT (CHAR_LENGTH(considerations)=0)" \
                    "AND (chamber IS NOT NULL)" \
                    "ORDER BY CHAR_LENGTH(facts)").format(sql.Identifier(language))

    with psycopg2.connect(config) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            while True:
                rows = cursor.fetchmany(1000)
                if not rows:
                    break
                for row in rows:
                    idx, chamber, date, html_url, pdf_url, facts, considerations, rulings, judgements = row
                    if str(html_url) != "":
                        link = html_url
                    if str(pdf_url) != "":
                        link = pdf_url
                    res = {"id": idx, "text": facts, "legal_area": get_legal_area(chamber), "date": int(date.year),
                           "link": str(link), "considerations": considerations, "ruling": rulings,
                           "judgment": judgements}
                    link = ""
                    if len(res["judgment"]) == 1 and res["legal_area"] in LEGAL_AREAS:
                        unfiltered_dataset.append(res)

            return filter_dataset(unfiltered_dataset)


if __name__ == '__main__':
    usage = "Arguments:\n - language: the language you want to extract the dataset from, e.g. the relation de. " \
            "Type -all to create dataset for all the possible languages. \n " \
            "Usage:\n python datasetCreation.py language\n" \
            "Example:\n python datasetCreation.py de"
    try:
        assert len(sys.argv) == 2
        try:
            if sys.argv[1] == "-all":
                for l in ["de", "fr", "it"]:
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
