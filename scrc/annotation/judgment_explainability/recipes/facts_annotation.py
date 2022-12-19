import prodigy
import spacy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens

PORTS = {"de": 11000, "fr": 12000, "it": 13000}
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]

"""
This recipe loads the facts, judgment and link to ruling from the dataset.
The recipe serves the data to the prodigy app, allowing to annotate the facts by considering the consideration and judgment.
Run with prodigy facts-annotation language -F ./judgment_explainability/recipes/facts_annotation.py
"""


@prodigy.recipe(
    "facts-annotation",
    language=("The language to use for the recipe.", 'positional', None, str),
)
# function called by the @prodigy-recipe definition
def facts_annotation(language: str):
    nlp = spacy.load(f"{language}_core_news_sm")  # Load the spaCy model for tokenization.
    stream = JSONL(
        f"./judgment_explainability/annotation_datasets/annotation_input_set_{language}.jsonl")  # Input dataset
    dataset = f"annotations_{language}"  # Outpu dataset
    port = PORTS[language]

    # Tokenize the incoming examples and add a "tokens" property to each example.
    stream = add_tokens(nlp, stream, use_chars=None)

    return {
        "dataset": dataset,  # Name of dataset to save annotations
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
                {"view_id": "spans_manual", "lang": nlp.lang, "labels": LABELS},
                {"view_id": "text_input", "field_label": "Annotator comment on this ruling",
                 "field_placeholder": "Type here...", "field_rows": 5},
            ]
        },

    }
