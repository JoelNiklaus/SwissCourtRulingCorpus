# README
This project is part of the Bachelor Thesis __Using Occlusion to explain Legal Judgment Prediction in Switzerland__.
It sets up prodigy using Docker and is an adaption of the `judgment_outcome` task. 
The prodigy setup contains the [facts_annotation recipe](recipes/facts_annotation.py) and 
[inspect_facts_annotation recipe](recipes/inspect_facts_annotation.py). The datasets for the annotation (de, fr, it) 
where created with the [prodigy_dataset_creation](scrc/annotation/prodigy_dataset_creation/prodigy_dataset_creation.py).
The test set for the occlusion experiment can be created using the [experiment_creato](scrc/annotation/judgment_explainability/occlusion/experiment_creator.py)
script. This project also contains tools to analyse the annotation and the results from the occlusion experiments.

```
project structure
...
├── scrc
│   └─── annotation
│       ├── judgment_explainability
│       │   ├── analysis (Contains scripts for analysis)
│       │   ├── guidelines (Contains pdfs and tex of guidelines)
│       │   ├── occlusion (Contains experiment_creator script)
│       │   └── recipes (Contains prodigy recipes)
│       │
│      ...
│       └── prodigy_dataset_creation (Contains the script to create the annotaion datasets)
│
...

Note some directories must be created in particular judgment_explainability/legal_expert_annotations and the different data
directory referenced in gitignore.
```




## Facts annotation for Legal Judgment Prediction in Switzerland
For this project, the prodigy tool is used to gather annotations and subsequently explanations from legal experts to create ground truths. 
These ground truths could give insight into the inner workings of the model used for the prediction in the Swiss Court Ruling Corpus 
presented by _Niklaus, Chalkidis, and Stürmer (2021)_.
The facts are provided to prodigy by a JSONL file which can bex created by running the [prodigy_dataset_creation](scrc/annotation/prodigy_dataset_creation/prodigy_dataset_creator.py).
Note that the input files produced by the ``prodigy_dataset_creator`` can change when running the script again (when the database changes).
It is therefore advised to safe a copy of your files after you started the annotation, so that your input dataset remains the same.
The annotated data is saved in the annotations.db.
The `facts_annotation` recipe allows the annotation of the fact section of a court ruling. It uses prodigy's block 
interface to display the span_manual task as well as the link and title of the ruling and a free text comment section.
The `inspect_facts_annotation` allows the annotator to go back and revise 
the annotations according to the newest guidelines. The guidelines for this annotation task can be found in the `guidelines` directory.
To create gold standard annotations' prodigy's inbuilt [review recipe](https://prodi.gy/docs/recipes#review) is used.
To run a recipe please follow the directions given in `setup.sh` and `run.sh`. The different tasks use different ports which are 
specified in the different recipes. Note that because we use a built-in recipe for the gold standard annotations only one gold standard task can be run at the time.
The other recipes can be run in parallel to each other.

## Occlusion for Legal Judgment Prediction in Switzerland
@ToDo
### Test set creation
@ToDo
### Running on Ubelix
@ToDo

## Refrences
Niklaus, J., Chalkidis, I., & St ̈urmer, M. (2021). Swiss-judgment-prediction: A multilingual legal
judgment prediction benchmark. arXiv. Retrieved from https://arxiv.org/abs/2110.00806
DOI: 10.48550/ARXIV.2110.00806