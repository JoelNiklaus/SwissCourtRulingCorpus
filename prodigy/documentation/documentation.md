# Documentation – Explainability Annotations for Legal Judgment Prediction in Switzerland
[Project link](https://www.digitale-nachhaltigkeit.unibe.ch/studies/bachelor_s__and_master_s_theses_at_inf/natural_language_processing/explainability_annotations_for_legal_judgment_prediction_in_switzerland/index_eng.html)
## Introduction
Swiss court decisions are anonymized to protect the privacy of the involved people (parties, victims, etc.). Previous research [1] has shown that it is possible to re-identify companies involved in court decisions by linking the rulings with external data in certain cases.
Our project tries to further build an automated system for re-identifying involved people from court rulings.
This system can then be used as a test for the anonymization practice of Swiss courts. For more information regarding the overarching research project, please go here.

- We recently presented a dataset for Legal Judgment Prediction (you try to predict the outcome of a case based on its facts) including 85K Swiss Federal Supreme Court decisions [2].
- Although we achieved up to 80% Macro-F1 Score, the models still work as black boxes and are thus not interpretable.
- In this project, you will work together with a legal student, to annotate some court decisions in the dataset for explainability.
- You will use the annotation tool [Prodigy](https://prodi.gy/) and write annotation guidelines for collecting high-quality annotations.

## Research Question

So far, to the best of our knowledge, the Explainable AI methods have not been studied in Swiss Legal Judgment Prediction.

__RQ1: To what extent do current models predict the judgment outcome based on input deemed important by lawyers?__

__Steps__
1. Get familiar with the literature on explainable NLP.
2. Write the annotation guidelines.
3. Set up Prodigy for the annotation process.
4. Work together with the law student to collect the annotations.
5. Analyze the annotated data and run experiments based on the annotations.
6. Describe the annotation process and the annotated data in detail

## Organisation
### GitHub
This project is a fork of [the Swiss Court Ruling Corpus](https://github.com/JoelNiklaus/SwissCourtRulingCorpus.git) and further develops the prodigy setup of @Skatinger described in [this branch](https://github.com/Skatinger/SwissCourtRulingCorpus/tree/prodigy).
The current branch on which this project is developed is called prodigy.
To check if the branch is correct use:
```shell
git branch --show-current
```
If you're happy with your changes add your files and make a commit to record them locally:
```shell
git add <FILE YOU MODIFIED>
git commit
```
For good commit messages use for example [the following guide](https://chris.beams.io/posts/git-commit/)

It is a good idea to sync your copy of the code with the original repository regularly. This way you can quickly account for changes:
```shell
git fetch upstream
git rebase upstream/main
```

### Sandbox Access
To access the sandbox use ``ssh nina@fdn-sandbox3.inf.unibe.ch`` and your ssh key. The sandbox was added as a remote host to the pyCharm application and can be browsed via Tools -> Deployment -> Browse Remote host. For this project you must run your prodigy recepies on the sandbox.

### DB Access
Access the SCRC DB from sandbox using

```shell
psql -h localhost -p 5432 -U readonly -d scrc
```
and the credential defined in your ``.env`` file.
On scrc you can select all spiders using the SQL query ``SELECT DISTINCT spider FROM de;``

```

            spider
-------------------------------
 CH_BGE
 LU_Gerichte
 ZH_Verwaltungsgericht
 NW_Gerichte
 ZH_Sozialversicherungsgericht
 VD_FindInfo
 AR_Gerichte
 ZH_Baurekurs
 BE_Verwaltungsgericht
 BE_BVD
 GL_Omni
 SG_Publikationen
 SG_Gerichte
 CH_WEKO
 BL_Gerichte
 BE_Weitere
 CH_BVGer
```
### How to chose cases from DB - Composition of the annotation dataset
#### Questions:
- Only choose cases from validation and test set ==> How do I know which is which in the scrc:
    * test cases have years: 2017 - 2020
    * validation cases have years: 2015 - 2016
- Only select judgments where the model has high certainty and is right ==> where do i see this @joel
- balanced dataset: 50% approval and dismissal (how many cases __1000>__)
    - We have 4 dimensions:
        - 3 Languages (DE, FR, IT)
        - We choose 3 law areas
        - 2 outcomes (approval, dismissal)
        - 7 years, we take the same amount of cases from each year, if we distribute the cases from 2015 - 2021 (=> 7 year)
    ==> One ruling per dimensions 3 (language) * 3 (law area) * 2 (outcome) * 6 (year) * 1 (number of rulings) = 108 cases
- Not all legal areas: penal law and other law area where model was second best (social law), maybe also include civil law ==> Which areas
- On the legal area, we talked about penal law which others (aka where was the model second best)? And how do I find these cases in the DB? (Do you have a list of spiders or a certain column where this is defined (I searched the DB and did not find a good answer)?
In main_utils.py you find a function get_legal_area which helps
- Take easy cases: short and easy cases: ask Thomas if the students can have an opinion about it (make a facts limit?-> sort from short to long)

### Prodigy Setup for the annotations
- Let law students annotate the sentences alias the facts (we can later expand it to paragraphs if needed)
- Give them the entire ruling with considerations, so that it makes it easier for students to annotate correctly

## Interpretation of results
for entire annotations with both law students ==> calculate inter annotator agreement

### Experiments
- special losses
- occlusion tests (if I delete certain part of text, how does it change the output)
==> ask Ilias later again how to do this

## Docker for Dummies

Check if container `prodigy_v1_nina` is running:
```shell
docker container ls -a
``
Run the following command to remove Docker container:
```shell
docker stop <Container_ID>
docker rm <Container_ID>
```
Run `bash setup.sh` in the `/prodigy` folder. This builds the docker image, starts the container and prepares everything.
everything.
To run a task, use `run.sh` in the `/prodigy` folder. Pass it the desired parameters explained atop the file. It starts
your task as a webserver. Access it at http://fdn-sandbox3.inf.unibe.ch:8080/. Login with the credentials configured in the Dockerfile ("admin": "password")
If your changes don't seem to have an impact you might want to rebuild the image via
```shell
build -t prodigy_v1_nina .
```
### Workflow
- If you change something in a recipe and want to test it use ``bash develop.sh`` and then the appropriate ``prodigy``command.
- For the changes to take to the actual docker setup you will have to rebuild the image as explained above
- Work on server via pycharm and sync files to local disk.
- Commit from local disk via PyCharm Github Tool

## ToDos
- Please refrence https://docs.google.com/spreadsheets/d/16as87DkqozifgGYGFfWNR7o0Un1rd2jBqSYz6C-9eq0/edit#gid=0
a list will be added here.
- [] For now the results are dumped into annotations.db, find way to dump them in the scrc db
- [] Get the correct cases which are not yet annotated and fulfill all the requirements discussed above


