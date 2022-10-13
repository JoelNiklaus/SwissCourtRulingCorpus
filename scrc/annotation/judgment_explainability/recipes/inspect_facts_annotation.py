import prodigy
from prodigy.components.loaders import JSONL

PORTS ={"angela_de":11001, "lynn_de":11002, "thomas_de":11003, "lynn_fr": 12001, "lynn_it": 13001}
LABELS = ["Supports judgment", "Opposes judgment", "Lower court"]

"""
Docs
This recipe loads the facts, judgment and link to ruling from the dataset.
The recipe serves the data to the prodigy app, allowing to revise the annotations according to the newest guidelines.
Run with prodigy inspect-facts-annotation language annotator -F ./judgment_explainability/recipes/facts_annotation.py
"""


# the recipe itself, for info on how to use it, see the prodigy docs
@prodigy.recipe(
  "inspect-facts-annotation",
    language=("The language to use for the recipe. Type test for test mode", 'positional', None, str),
    annotator=("The annotator who marking the cases.", 'positional', None, str),
)

# function called by the @prodigy-recipe definition
def inspect_facts_annotation(language:str,annotator:str ):

  stream = JSONL(f"./judgment_explainability/legal_expert_annotations/{language}/annotations_{language}-{annotator}.jsonl")
  dataset = f"annotations_{language}_inspect"
  port = PORTS[f"{annotator}_{language}"]

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
        {"view_id": "spans_manual", "labels": LABELS},
        {"view_id": "text_input","field_label":"Annotator comment on this ruling", "field_placeholder": "Type here...","field_rows": 5},
      ]
    },

    }

