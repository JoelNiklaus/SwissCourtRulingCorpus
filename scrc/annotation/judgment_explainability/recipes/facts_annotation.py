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
from prodigy.components.db import connect
# prodigy facts-annotation de "" test -F recipes/facts_annotation.py
ports ={"de_1":11001,"de_2":11002,"de_3":11003, "fr_1": 12000, "it_1": 13000 }
## Docs
#
# This recipe is used to load facts, considerations, rulings and judgment directly from the scrc database configured in the environment variables.
# The recipe serves the data to the prodigy app, allowing to annotate the facts by considering the consideration and judgment.

# helper function for recipe command line input
def split_string(string):
  return string.split(",")

# the recipe itself, for info on how to use it, see the prodigy docs
@prodigy.recipe(
  "facts-annotation",
  language=("The language to use for the recipe", 'positional', None, str),
  annotator=("The annotator of the set", 'positional', None, str),
  test_mode=("Running in test mode",'positional', None, str )
)

# function called by the @prodigy-recipe definition
def facts_annotation(language:str,annotator:str,test_mode:str ):
  # Load the spaCy model for tokenization.
  # might have to run "python -m spacy download de_core_news_sm" @Todo
  # Have to load french and italian model
  nlp = spacy.load("{}_core_news_sm".format(language))
  # define labels for annotation
  labels = ["Supports judgment","Opposes judgment"]

  if annotator != "" and test_mode != "test" :
    dataset = "annotations_{}_{}".format(language,annotator)
    port = ports["{}_{}".format(language,annotator)]
    stream = JSONL("./datasets/annotation_input_set_{}.jsonl".format(language))
  else:
    dataset = "annotations_{}_test".format(language)
    port = 14000
    stream = JSONL("./datasets/annotation_input_set_{}_ex.jsonl".format(language))


  # define stream (generator)


  # Tokenize the incoming examples and add a "tokens" property to each
  # example. Also handles pre-defined selected spans. Tokenization allows
  # faster highlighting, because the selection can "snap" to token boundaries.
  # If `use_chars` is True, tokens are split into individual characters, which enables
  # character based selection as opposed to default token based selection.
  stream = add_tokens(nlp, stream, use_chars=None)
  return {
    "dataset": dataset ,# Name of dataset_scrc to save annotations
    "view_id": "blocks",
    "stream": stream,
    "config": {
      "port": port,
      "blocks": [
        {"view_id": "html",
         "html_template": "<p style='float:left'>{{file_number}}</p>"},
        {"view_id": "html", "html_template": "<h1 style='float:left'>{{header}} – Judgment: {{judgment}}</h2>"},
        {"view_id": "html",
         "html_template": "<h2 style='float:left'>Facts</h2><a style='float:right' href='{{link}}'>Got to the Court Ruling</a>"},
        {"view_id": "spans_manual", "lang": nlp.lang, "labels": labels},
        {"view_id": "text_input","field_label":"Annotator comment on this ruling", "field_placeholder": "Type here...","field_rows": 5},
      ]
    },

    }

