import sqlite3
import spacy
import json
import prodigy
from prodigy.components.loaders import JSONL
from prodigy.util import split_string
from prodigy.components.preprocess import add_tokens
import psycopg2
from psycopg2 import sql
from typing import List, Optional
import os

# prodigy facts-annotation de -F recipes/facts_annotation.py

## Docs
#
# This recipe is used to load facts, considerations, rulings and judgment directly from the scrc database configured in the environment variables.
# The recipe serves the data to the prodigy app, allowing to annotate the facts by considering the consideration and judgment.
#
# Basic SQL Query:
# SELECT id,facts,considerations,rulings,judgements FROM de
# WHERE NOT (facts like '' OR considerations like '' OR rulings like '' OR judgements IS NULL);


# a generator which returns records from the database, using batched fetching
# format is for the `choice` view of the prodigy app
def db_stream(lng):
  # scrc database connection configuration string
  config = "dbname=scrc user={} password={} host=localhost port=5432".format(os.environ["DB_USER"], os.environ["DB_PASSWORD"])

  # prodigy database connection configuration string, to fetch processed ids
  # TODO:
  # - remove hardcoded login credentials
  # - find the ids which have been labeled already
  # - check which records have already been labeled
  # last two are easier done once the annotations are direclty fed to the scrc database, allowing for a fast query
  # instead of parsing all the stored data from prodigy
  latest_id = 0
  ids_done = []


  # postgres is much faster parsing VALUES() to relations than using a NOT IN query
  # use sql.Identifier and parameters to prevent sql injections
  # SELECT id,facts,considerations,rulings,judgements FROM de
  # WHERE NOT (facts like '' OR considerations like '' OR rulings like '' OR judgements IS NULL);
  '''
  query = sql.SQL("SELECT id,facts,considerations,rulings,judgements " \
          "FROM {}" \
          "WHERE NOT (facts like '' OR considerations like '' OR rulings like '' OR judgements IS NULL) " \
          "AND spider like spider "        
          "AND id > %s ").format(sql.Identifier(language))
  '''
  query = sql.SQL("SELECT id,facts " \
                  "FROM de " \
                  "WHERE facts NOT like '' AND spider like 'CH_BGer'"
                  "AND id > %s ").format(sql.Identifier(lng))
  params = (latest_id,)
  # if any judgments have been labeled already, filter them out
  if len(ids_done) > 0:
    ids_done = ", ".join(["({})".format(id) for id in ids_done]) # format for sql query
    query += " AND id NOT IN (VALUES(%s);"
    params += (ids_done,)
  
  # TODO: ensure the connection is closed and reopened on every generator call in case problems with the connection arise
  with psycopg2.connect(config) as connection:
    with connection.cursor() as cursor:
      cursor.execute(query, params)
      while True:
        rows = cursor.fetchmany(1000)
        if not rows:
          break
        for row in rows:
          idx, facts = row
          res = {"id": idx, "text": facts}
          res = json.dumps(res)

          if len(res) > 0:
            yield res

# helper function for recipe command line input
def split_string(string):
  return string.split(",")

  def update(examples):
    # This function is triggered when Prodigy receives annotations
    print(f"Received {len(examples)} annotations!")


# the recipe itself, for info on how to use it, see the prodigy docs
@prodigy.recipe(
  "facts-annotation",
  language=("The language to use for the recipe", 'positional', None, str)
)

# function called by the @prodigy-recipe definition
def facts_annotation(language:str):
  # Load the spaCy model for tokenization.
  # might have to run "python -m spacy download de_core_news_sm" @Todo
  nlp = spacy.load("de_core_news_sm")
  # define labels for annotation
  labels = ["Supports judgment","Opposes verdict"]

  dataset = "annotations_"+language

  # define stream (generator)
  stream = JSONL(db_stream(language))

  # Tokenize the incoming examples and add a "tokens" property to each
  # example. Also handles pre-defined selected spans. Tokenization allows
  # faster highlighting, because the selection can "snap" to token boundaries.
  # If `use_chars` is True, tokens are split into individual characters, which enables
  # character based selection as opposed to default token based selection.

  stream = add_tokens(nlp, stream, use_chars=None)




  return {
    "dataset": dataset ,# Name of dataset to save annotations
    "view_id": "spans_manual",
    "stream": stream,
    "config": {  # Additional config settings, mostly for app UI
      "lang": nlp.lang,
      "labels": labels,  # Selectable label options
    }
  }
