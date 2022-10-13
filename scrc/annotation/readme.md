# README
The prodigy tool is used to annotate data manually. Please reference the readmes for the individual task for further information.
Currently, three task are implemented.

## Setup & Usage
1. Request the file `.env` from a maintainer *OR* create your own, with the format:
```bash
DB_USER=<your db username here>
DB_PASSWORD=<your database password here>
```
Place it under `./annotation`
2. Create the output database for the prodigy task and change the configuration in ``prodigy.json`` to fit your setup.
Currently, the setup the db is the following:

```json
{
 "db": "sqlite",
  "db_settings": {
    "sqlite": {
      "name": "prodigy_results.db",
      "path": "./"
    }
  }
}
```
3. Create the input datasets using the dataset_creation script at ```./prodigy_dataset_creation/prodigy_dataset_creator.py```.
Note that you need to request the sjp datasets from @JoelNiklaus (see `prodigy_dataset_creator.py` for exact files).
4. If you are not in the docker group, request access from a maintainer to be able to run docker commands.
5. Run `bash setup.sh`. This builds the docker image, starts the container and prepares
everything.
6. To run the task, use `run.sh` with your chosen task and the desired parameters. It starts
your task as a webserver. Login with the credentials specified in the ``.env`` file.

## Tasks
The judgment explainability and the prediction task need their own dataset which can be produced using the scripts in ```prodigy_dataset_creation/```
For more information on how to run the task please consult the documentation in ``run.sh``.
### Judgment explainability

Contains the `facts_annotation recipe` and `inspect-facts_annotation recipe`, which allows to annotate the facts of a ruling using a span_manual task.

### Judgment prediction

Contains the ``judgment_prediction recipe``, which allows to choose the judgment for the given facts.

### Judgment outcome

Contains the ``judgment_outcome recipe``, which enables text classification (judgment extraction).
__Note that this recipe is no longer maintained and has to be run with ```./judgement_outcome/jo_run.sh```__
## Development

To develop your task or add some changes to existing tasks (recipes), use the script `develop.sh`. (Run setup.sh) beforehand
if not yet done, the base image has to exist. The develop script will start a development session using the files on the
host machine, allowing for easy editing. Run your recipe directly within the development shell to test around.
Don't forget to run `setup.sh` again when you are done, to apply your changes to the productive version of the application.
For more details see the docs in the script `bash develop.sh`.

## Configuration

If you would like to change configurations edit the file `prodigy.json` and then rebuild and restart the service
by following the steps in **Setup & Usage** or test around in the development environment. Please note that some configurations may be added in the recipe itself.

## Custom Usage

To create your own module write a recipe as explained in the prodi.gy docs, or extend one of theirs. Use the `develop.sh` script
for easy access and testing.

Prodigy documentation: https://prodi.gy/docs/