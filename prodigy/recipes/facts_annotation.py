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

import ujson
from pathlib import Path

def read_jsonl(file_path):
    """Read a .jsonl file and yield its contents line by line.
    file_path (unicode / Path): The file path.
    YIELDS: The loaded JSON contents of each line.
    """
    with Path(file_path).open('r', encoding='utf8') as f:
        for line in f:
            try:  # hack to handle broken jsonl
                yield ujson.loads(line.strip())
            except ValueError:
                continue


# helper function for recipe command line input
def split_string(string):
  return string.split(",")

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
  stream = JSONL("datasets/annotation_input_set_"+language+".jsonl")

  # Tokenize the incoming examples and add a "tokens" property to each
  # example. Also handles pre-defined selected spans. Tokenization allows
  # faster highlighting, because the selection can "snap" to token boundaries.
  # If `use_chars` is True, tokens are split into individual characters, which enables
  # character based selection as opposed to default token based selection.
  stream = add_tokens(nlp, stream, use_chars=None)

  return {
    "dataset": dataset ,# Name of dataset to save annotations
    "view_id": "blocks",
    "stream": stream,
    "config": {
      "blocks": [
        {"view_id": "html", "html": "<h2 style='float:left'>Facts</h2><a style='float:right' href='{{link}}'>Got to the Court Ruling</strong>"},
        {"view_id": "spans_manual", "lang": nlp.lang, "labels": labels},
        {"view_id": "html", "html_template": "<h2 style='float:left'>Considerations</h2>"},
        {"view_id": "html", "html_template":"<p style='hyphens: auto;   text-align: justify'>{{considerations}}</>"},
        {"view_id": "html", "html_template":"<h2 style='float:left'>Ruling</h2><br>"},
        {"view_id": "html", "html_template":"<p style='hyphens: auto;   text-align: justify'>{{ruling}}</p>"}
        # Explain your decision field? {"view_id": "text_input", "field_rows": 3},
      ]
    }
  }
