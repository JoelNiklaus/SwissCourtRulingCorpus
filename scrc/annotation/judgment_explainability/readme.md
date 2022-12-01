# README

This project is part of the Bachelor Thesis __Using Occlusion to explain Legal Judgment Prediction in Switzerland__. It
sets up prodigy using Docker and is an adaption of the `judgment_outcome` task. The prodigy setup contains
the [facts_annotation recipe](recipes/facts_annotation.py) and
[inspect_facts_annotation recipe](recipes/inspect_facts_annotation.py). The datasets for the annotation (de, fr, it)
where created with the [prodigy_dataset_creation](scrc/annotation/prodigy_dataset_creation/prodigy_dataset_creation.py).
The test set for the occlusion experiment can be created using
the [experiment_creato](scrc/annotation/judgment_explainability/occlusion/experiment_creator.py)
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

For this project, the prodigy tool is used to gather annotations and subsequently explanations from legal experts to
create ground truths. These ground truths could give insight into the inner workings of the model used for the
prediction in the Swiss Court Ruling Corpus presented by _Niklaus, Chalkidis, and Stürmer (2021)_. The facts are
provided to prodigy by a JSONL file which can bex created by running
the [prodigy_dataset_creation](scrc/annotation/prodigy_dataset_creation/prodigy_dataset_creator.py). Note that the input
files produced by the ``prodigy_dataset_creator`` can change when running the script again (when the database changes).
It is therefore advised to safe a copy of your files after you started the annotation, so that your input dataset
remains the same. The annotated data is saved in the annotations.db. The `facts_annotation` recipe allows the annotation
of the fact section of a court ruling. It uses prodigy's block interface to display the span_manual task as well as the
link and title of the ruling and a free text comment section. The `inspect_facts_annotation` allows the annotator to go
back and revise the annotations according to the newest guidelines. The guidelines for this annotation task can be found
in the `guidelines` directory. To create gold standard annotations' prodigy's
inbuilt [review recipe](https://prodi.gy/docs/recipes#review) is used. To run a recipe please follow the directions
given in `setup.sh` and `run.sh`. The different tasks use different ports which are specified in the different recipes.
Note that because we use a built-in recipe for the gold standard annotations only one gold standard task can be run at
the time. The other recipes can be run in parallel to each other.

## Occlusion for Legal Judgment Prediction in Switzerland

@ToDo

### Test set creation

@ToDo

### Running on Ubelix

Some part of the implementation can take very long to run. If you have access to a high-performance computing cluster we
recommend these parts on there. These instructions refer to UBELIX the cluster of the university of Berne. It is
recommended to first read the [documentation](https://hpc-unibe-ch.github.io/quick-start.html)
to get familiar with the infrastructure.

If you are already familiar with UBELIX please follow the following steps:

1. Open the .bashrc file in your $HOME Folder and enter module load CUDA
2. Enter `module load Anaconda3` in the terminal
3. Enter the conda environment using `eval "$(conda shell.bash hook)"`

#### Occlusion

This implementation of the occlusion method has the aim to produce new prediction with the sjp model using 
the occlusion test sets created  with ``experiment_creator.py``. To run the occlusion experiments on Ubelix. Clone 
the  SwissJudgementPrediction repository into your home directory using ``git clone https://github.com/JoelNiklaus/SwissJudgementPrediction.git``. 
1. If not already done create a directory called data with a subdirectory for each language you want to test. Place `train.csv`,
`val.csv` and your desired occlusion test set under each language directory
2. Change the file name in `SwissJudgementPrediction/run_tc.py` from `test.csv` to the name of your occlusion test set.
3. Create a new environment called "sjp" and install packages from the env.yml file using `conda env create -f env.yml`
4. Activate the sjp environment using `conda activate sjp`.
5. Use the following command to install the right version of PyTorch: `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`
6. Create a Weights & Biases account, get your API token, and enter wandb login inside your conda environment. 
After you entered the token, it will be saved in the .netrc file in you $HOME folder
7. Run occlusion using ``sbatch run_ubelix_job.sh adapters test xlm-roberta-base hierarchical lang lang switzerland no_augmentation civil_law False``

The prediction results will be placed under ``/SwissJudgementPrediction/sjp/adapters/xlm-roberta-base-hierarchical/lang/2``.
Note that you might need to adapt some line in ``run_ubelix_job.sh`` to fit your setup.


#### Analysis

The annotation_analysis uses BERT (for the bert_score). Please follow the general steps above before running the
analysis. Then clone this repository into your home directory
using ``git clone https://github.com/ninabaumgartner/SwissCourtRulingCorpus.git``. Change to the analysis directory
with ``cd SwissCourtRulingCorpus/scrc/annotation/judgment_explainability/analysis``.

1. For each py file in the analysis directory change the import path
   from `scrc.annotation.judgment_explainability.analysis.utils.module as module`
   To `utils.module as module`.
3. Create a new environment called "judgment-explainability" with ``conda env create -f env.yml``
   3.Activate the "judgment-explainability" environment using ``conda activate judgment-explainability``
4. Place your annotation under directory ``legal_expert_annotation/language/`` using the ``scp`` command.
5. Run the analysis using ``sbatch run_analysis_ubelix.sh``
   Note that if you want to use another file structure or naming convention the paths in the scripts have to adapted
   accordingly.

## Refrences

Niklaus, J., Chalkidis, I., & St ̈urmer, M. (2021). Swiss-judgment-prediction: A multilingual legal judgment prediction
benchmark. arXiv. Retrieved from https://arxiv.org/abs/2110.00806
DOI: 10.48550/ARXIV.2110.00806