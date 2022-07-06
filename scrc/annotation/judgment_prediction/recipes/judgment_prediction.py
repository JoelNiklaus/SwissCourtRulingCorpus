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
ports ={"de":14000,"fr": 15000, "it": 16000}
facts ={"de":"Sachverhalt", "fr":"Faits", "it":"Fatti"}
## Docs
#
# This recipe is used to load facts from the dataset.
# The recipe serves the data to the prodigy app, allowing to chose the appropriate judgment for the given facts.

# helper function for recipe command line input
def split_string(string):
  return string.split(",")


def get_stream(stream):
    while True:
      for task in stream:
        yield task


def add_options(stream):
    # Helper function to add options to every task in a stream
    options = [
        {"id": "happy", "text": "😀 happy"},
        {"id": "sad", "text": "😢 sad"},
        {"id": "angry", "text": "😠 angry"},
        {"id": "neutral", "text": "😶 neutral"},
    ]
    for task in stream:
        task["options"] = options
        yield task


# the recipe itself, for info on how to use it, see the prodigy docs
@prodigy.recipe(
  "judgment-prediction",
  language=("The language to use for the recipe.", 'positional', None, str),
)

# function called by the @prodigy-recipe definition
def judgment_prediction(language:str ):

    # stream = add_options(stream)

  # Load the spaCy model for tokenization.
  nlp = spacy.load("{}_core_news_sm".format(language))
  stream = JSONL("./judgment_explainability/datasets/annotation_input_set_{}.jsonl".format(language))

  dataset = "judgment_predictions_{}".format(language)
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
           "html_template": "<h2 style='float:left'>{}</h2>".format(facts[language])},
          {"view_id": "html",
           "html_template": "<p style='float:left'>{{text}}</p>"},
          {"view_id": "choice", "text": "Please chose the appropriate judgment for the facts above"},
          {"view_id": "text_input", "field_label": "Explain your decision", "field_placeholder": "Type here...",
           "field_rows": 5},]
    },

    }

