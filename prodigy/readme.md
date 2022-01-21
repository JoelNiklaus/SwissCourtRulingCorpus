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
3. Just run `bash install.sh` in the `/prodigy` folder. This builds the docker image, starts the container and prepares
everything.

4. To run a task, use `run.sh` in the `/prodigy` folder. It starts the judment extraction classification server. Access it
at http://fdn-sandbox3.inf.unibe.ch:8080/. Login with the credentials configured in the Dockerfile ("admin": "password")


## Configuration
If you would like to change configurations edit the file `prodigy.json` and then rebuild and restart the service
by following the steps in **Setup & Usage**.

config everything in prodigy.json:
- database connection
- host: use 0.0.0.0 if deployed in docker container, otherwise localhost (except if db is remote)


## Custom Usage
To create your own module write a recipe as explained in the prodi.gy docs, or extend one of theirs.

Prodigy documentation: https://prodi.gy/docs/

To edit and test your module you have two options:

1) Write it outside the container and mount the file manually, then run it.
2) Execute a bash shell in the container and use terminal tools to write and test your file.

See docker docs for further infos.