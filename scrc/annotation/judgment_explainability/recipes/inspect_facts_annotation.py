#  prodigy inspect-facts-annotation de nina -F ./recipes/inspect_facts_annotation.py

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
ports ={"angela":11001, "lynn":11002, "thomas":11003,"fr": 12001, "it": 13001}

# helper function for recipe command line input
def split_string(string):
  return string.split(",")

# the recipe itself, for info on how to use it, see the prodigy docs
@prodigy.recipe(
  "inspect-facts-annotation",
    language=("The language to use for the recipe. Type test for test mode", 'positional', None, str),
    annotator=("The annotator who marking the cases.", 'positional', None, str),
)

# function called by the @prodigy-recipe definition
def inspect_facts_annotation(language:str,annotator:str ):
  # define labels for annotation
  labels = ["Supports judgment","Opposes judgment","Lower court"]
  stream = JSONL("./annotations/annotations_{}-{}.jsonl".format(language,annotator))


  dataset = "annotations_{}_inspect".format(language)
  port = ports[annotator]

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
         "html_template": "<h2 style='float:left'>Facts</h2><a style='float:right' href='{{link}}' target='_blank'>Go to the court ruling</a>"},
        {"view_id": "spans_manual", "labels": labels},
        {"view_id": "text_input","field_label":"Annotator comment on this ruling", "field_placeholder": "Type here...","field_rows": 5},
      ]
    },

    }

