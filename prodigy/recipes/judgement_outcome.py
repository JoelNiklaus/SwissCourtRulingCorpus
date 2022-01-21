import prodigy
import psycopg2
from typing import List
import os

# a generator which returns records from the database, using batched fetching
# format is for the `choice` view of the prodigy app
def db_stream():
  config = "dbname=scrc user={} password={} host=localhost port=5432".format(os.environ["DB_USER"], os.environ["DB_PASSWORD"])
  latest_id = 0
  # TODO: make this configurable
  language = 'de'
  spider = 'BS_Omni'
  query = "SELECT id, spider, rulings FROM de WHERE spider = %s AND id > %s LIMIT 20"
  params = (spider, latest_id)

  with psycopg2.connect(config) as connection:
    with connection.cursor() as cursor:
      cursor.execute(query, params)
      while True:
        rows = cursor.fetchmany(100)
        if not rows:
          break
        for row in rows:
          idx, spider, ruling = row
          res = { "id": idx, "text": ruling }

          if len(res) > 0:
              yield res

# helper function for recipe command line input
def split_string(string):
  return string.split(",")

@prodigy.recipe(
  "judgment-outcome",
  dataset=("Name of dataset (spider)", "positional", None, str),
  options=("Labels to use for the dataset (comma separated)", "option", "o", split_string)
)

def judgment_outcome(dataset: str, options: List[str]):

  if not options:
      options = ["positiv", "negativ"]

  # define stream (generator)
  stream = db_stream()
  stream = add_options(stream, options)

  return {
    "dataset": dataset,
    "view_id": "choice",
    "stream": stream,
    "config": {
      "choice_style": "single",
      "choice_auto_accept:": "true"
    }
  }

def add_options(stream, options):
  options = [{"id": opt, "text": opt} for opt in options]
  for task in stream:
    task["options"] = options
    yield task
