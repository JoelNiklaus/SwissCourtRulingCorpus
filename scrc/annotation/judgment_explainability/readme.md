# README
This project is part of my Bachelor Thesis __Explainability Annotations for Legal Judgment Prediction in Switzerland__.
It sets up prodigy using Docker and is an adaption of the `judgment_outcome` task. 
It contains the [facts_annotation recipe](recipes/facts_annotation.py), which allows to annotate the facts of a ruling using a span_manual task. 
The datasets for each language (de, fr, it) where created with the [dataset_creation script](datasets/dataset_creation.py).

## Facts annotation for Legal Judgment Prediction in Switzerland
For this project the prodigy tool is used to gather annotations and subsequently explanation from legal experts to create ground truths. 
These ground truths could give insight into the inner workings of the model used for the prediction in the Swiss Court Ruling Corpus 
presented by _Niklaus, Chalkidis, and Stürmer (2021)_.
The facts are provided to prodigy by a JSONL file which can be created by running the [dataset_creation](/datasets/dataset_creation.py).
The annotated data is saved in the annotations.db.

The facts annotation recipe uses prodigy's block interface to display the span_manual task as well as the link and title of the ruling and a free text comment section. 
We are currently running 5 different facts annotation tasks. The different ports are specified in the recipe.

## Refrences
Niklaus, J., Chalkidis, I., & St ̈urmer, M. (2021). Swiss-judgment-prediction: A multilingual legal
judgment prediction benchmark. arXiv. Retrieved from https://arxiv.org/abs/2110.00806
DOI: 10.48550/ARXIV.2110.00806