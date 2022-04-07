# Overview
The prodigy tool is used to create ground truths by labelling data. Use this to quickly label your
data and using it as evaluation for your task.
##### Supported Tasks
- Text Classification (judment extraction)
- Named Entity Recognition (court composition extraction)
- Span Categorization (citation extraction)

So far only text classification for judment extraction is implemented.

## Setup & Usage
1. Request the file `.env` from a maintainer *OR* create it your own, with the format:
```bash
DB_USER=<your db username here>
DB_PASSWORD=<your database password here>
```
Place it under `./prodigy`
2. If you are not in the docker group, request access from a maintainer to be able to run docker commands
3. Just run `bash setup.sh` in the `/prodigy` folder. This builds the docker image, starts the container and prepares
everything.

4. To run a task, use `run.sh` in the `/prodigy` folder. Pass it the desired parameters explained atop the file. It starts
your task as a webserver. Access it
at _http://fdn-sandbox3.inf.unibe.ch:18080/_. Login with the credentials configured in the Dockerfile ("admin": "password")


## Development
To develop your task or add some changes to existing tasks (recipes), use the script `develop.sh`. (Run setup.sh) beforehand
if not yet done, the base image has to exist. The develop script will start a development session using the files on the
host machine, allowing for easy editing. Run your recipe directly within the development shell to test around.
Don't forget to run `setup.sh` again when you are done, to apply your changes to the productive version of the application.
For more details see the docs in the script `bash develop.sh`.

## Configuration
If you would like to change configurations edit the file `prodigy.json` and then rebuild and restart the service
by following the steps in **Setup & Usage** or test around in the development environment.

config everything in prodigy.json:
- database connection
- host: use _fdn-sandbox3.inf.unibe.ch_ if deployed in docker container, otherwise localhost (except if db is remote)

## Custom Usage
To create your own module write a recipe as explained in the prodi.gy docs, or extend one of theirs. Use the `develop.sh` script
for easy access and testing.

Prodigy documentation: https://prodi.gy/docs/
