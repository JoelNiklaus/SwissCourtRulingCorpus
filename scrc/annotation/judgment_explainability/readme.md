# README
This project is part of my Bachelor Thesis __Explainability Annotations for Legal Judgment Prediction in Switzerland__.
It sets up prodigy using Docker and is an adaption of the `judgment_outcome` task. 
It contains the [facts_annotation recipe](recipes/facts_annotation.py), which allows to annotate the facts of a ruling using a span_manual task. 
The datasets for each language (de, fr, it) where created with the [dataset_creation script](datasets/dataset_creation.py).

## Setup & Usage
1. Request the file `.env` from a maintainer *OR* create your own, with the format:
```bash
DB_USER=<your db username here>
DB_PASSWORD=<your database password here>
```
Place it under `./prodigy`
2. Create the output database for the prodigy task and change the configuration in [prodigy.json](prodigy.json) to fit your setup.
Currently, the set up the db is the following:

```json
{
  "db": "sqlite",
  "db_settings": {
    "sqlite": {
      "name": "annotations.db",
      "path": "./annotations"
    }
  }
}
```
3. Create the input datasets using the [dataset_creation script](datasets/dataset_creation.py). 
4. If you are not in the docker group, request access from a maintainer to be able to run docker commands.
5. Run `bash setup.sh` in the `/judgement_explainability` folder. This builds the docker image, starts the container and prepares
everything.
6. To run the task, use `run.sh` in the `/prodigy` folder. Pass it the desired parameters explained atop the file. It starts
your task as a webserver. Login with the credentials specified in the ``.env`` file.

## Development

To develop your task or add some changes to existing tasks (recipes), use the script `develop.sh`. (Run setup.sh) beforehand
if not yet done, the base image has to exist. The develop script will start a development session using the files on the
host machine, allowing for easy editing. Run your recipe directly within the development shell to test around.
Don't forget to run `setup.sh` again when you are done, to apply your changes to the productive version of the application.
For more details see the docs in the script `bash develop.sh`.

## Configuration
If you would like to change configurations edit the file `prodigy.json` and then rebuild and restart the service
by following the steps in **Setup & Usage** or test around in the development environment. Please note that some configurations may be added in the recipe itself.

## Facts annotation for Legal Judgment Prediction in Switzerland
For this project the prodigy tool is used to gather annotations and subsequently explanation from legal experts to create ground truths. 
These ground truths could give insight into the inner workings of the model used for the prediction in the Swiss Court Ruling Corpus 
presented by _Niklaus, Chalkidis, and Stürmer (2021)_.
The facts are provided to prodigy by a JSONL file which can be created by running the [dataset_creation](/datasets/dataset_creation.py).
The annotated data is saved in the annotations.db.

The facts annotation recipe uses prodigy's block interface to display the span_manual task as well as the link and title of the ruling and a free text comment section. 
We are currently running 5 different facts annotation tasks. The different ports are specified in the recipe.
 

### Custom Usage
To create your own module write a recipe as explained in the [Prodigy documentation](https://prodi.gy/docs/), or extend one an existing recipe. 
Use the `develop.sh` script for easy access and testing.

## Refrences
Niklaus, J., Chalkidis, I., & St ̈urmer, M. (2021). Swiss-judgment-prediction: A multilingual legal
judgment prediction benchmark. arXiv. Retrieved from https://arxiv.org/abs/2110.00806
DOI: 10.48550/ARXIV.2110.00806