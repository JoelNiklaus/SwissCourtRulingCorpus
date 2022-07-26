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
import re
ports ={"de":11000,"fr": 12000, "it": 13000}

## Docs
#
# This recipe loads the facts, judgment and link to ruling from the dataset.
# The recipe serves the data to the prodigy app, allowing to annotate the facts by considering the consideration and judgment.

# helper function for recipe command line input
def split_string(string):
  return string.split(",")


def get_stream(stream):
    while True:
      for task in stream:
        yield task

# the recipe itself, for info on how to use it, see the prodigy docs
@prodigy.recipe(
  "facts-annotation",
  language=("The language to use for the recipe.", 'positional', None, str),
)

# function called by the @prodigy-recipe definition
def facts_annotation(language:str ):
  # define labels for annotation
  labels = ["Supports judgment","Opposes judgment", "Lower court"]

  # Load the spaCy model for tokenization.
  nlp = spacy.load("{}_core_news_sm".format(language))
  stream = JSONL("./judgment_explainability/datasets/annotation_input_set_{}.jsonl".format(language))
  dataset = "annotations_{}".format(language)
  port = ports[language]

  # Tokenize the incoming examples and add a "tokens" property to each
  # example.
  stream = add_tokens(nlp, stream, use_chars=None)

  return {
    "dataset": dataset ,# Name of dataset to save annotations
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
        {"view_id": "spans_manual", "lang": nlp.lang, "labels": labels},
        {"view_id": "text_input","field_label":"Annotator comment on this ruling", "field_placeholder": "Type here...","field_rows": 5},
      ]
    },

    }

