# Contributors

- [Contributors](#contributors)
  - [Sandbox Setup](#sandbox-setup)
  - [Local setup](#local-setup)
  - [Developing](#developing)
    - [Before starting](#before-starting)
    - [Coding](#coding)
    - [Running a pipeline step](#running-a-pipeline-step)
    - [Opening a Pull Request](#opening-a-pull-request)
  - [Tasks](#tasks)
    - [Section Splitting](#section-splitting)
    - [Citation Extraction](#citation-extraction)
    - [Judgment Outcome Extraction](#judgment-outcome-extraction)
    - [Judicial Person Extraction](#judicial-person-extraction)
      - [Court Composition](#court-composition)
      - [Procedural Participation](#procedural-participation)
      - [Court Chambers](#court-chambers)
  - [Running long tasks](#running-long-tasks)
  - [SQL FAQ](#sql-faq)
    - [Joining Tables on queries with decision table](#joining-tables-on-queries-with-decision-table)
      - [Language](#language)
      - [File](#file)
    - [Joining the decision table on other tables](#joining-the-decision-table-on-other-tables)
    - [Where clause helper](#where-clause-helper)
      - [Checking for certain spiders](#checking-for-certain-spiders)
    - [Saving to database](#saving-to-database)
      - [Selecting file entries by file_name](#selecting-file-entries-by-file_name)
      - [Selecting file entries by file_name](#selecting-file-entries-by-file_name-1)
      - [Selecting decisions to delete](#selecting-decisions-to-delete)
    - [JOIN and map_join](#join-and-map_join)
      - [join](#join)
      - [map_join: Simple](#map_join-simple)
      - [map_join: With filling](#map_join-with-filling)
    - [Selecting "everything"](#selecting-everything)
      - [tables](#tables)
      - [fields](#fields)
      - [Setup database from scratch](#setup-database-from-scratch)
  - [Common Errors](#common-errors)
    - [PermissionError: [Errno 13] Permission denied: '/tmp/tika.log'](#permissionerror-errno-13-permission-denied-tmptikalog)
    - [idle_in_transaction_session_timeout](#idle_in_transaction_session_timeout)
  - [Questions?](#questions)

## Sandbox Setup

1. Fork the repository by clicking on the 'Fork' button on the repository's page. This creates a copy of the code under
   your GitHub user account.
2. Connect to the university vpn to be able to access the
   FDN-Server. [Tutorial](https://www.unibe.ch/university/campus_and_infrastructure/rund_um_computer/internetzugang/access_to_internal_resources_via_vpn/index_eng.html)
3. Connect to the FDN Server via ssh with the command

```bash
ssh <USERNAME>@fdn-sandbox3.inf.unibe.ch
```

4. In your home directory (default location after first login) you can clone your repository and then add the base
   repository as a remote

```bash
git clone git@github.com:<YOURGITHUBHANDLE>/SwissCourtRulingCorpus.git
cd SwissCourtRulingCorpus
git remote add upstream https://github.com/JoelNiklaus/SwissCourtRulingCorpus.git
```

5. Create a new branch with a descriptive change of your work, where you will make your changes:

```bash
git checkout -b a-descriptive-name-for-my-changes
```

6. Copy the progress files into your own directory. If you want to run a spider in a specific pipeline component, just
   delete that spider from the corresponding progress file.

```bash
cp -R /home/fdn-admin/SwissCourtRulingCorpus/data/progress ~/SwissCourtRulingCorpus/data/progress
```

7. Make a symlink to the spiders directory so that the AbstractPreprocessor can get the list of spiders.

```bash
ln -s /home/fdn-admin/SwissCourtRulingCorpus/data/spiders ~/SwissCourtRulingCorpus/data/spiders
```

## Local setup

1. Fork the repository by clicking on the 'Fork' button on the repository's page. This creates a copy of the code under
   your GitHub user account.
2. In your home directory (default location after first login) you can clone your repository and then add the base
   repository as a remote

```bash
git clone git@github.com:<YOURGITHUBHANDLE>/SwissCourtRulingCorpus.git
cd SwissCourtRulingCorpus
git remote add upstream https://github.com/JoelNiklaus/SwissCourtRulingCorpus.git
```
3. Create a new branch with a descriptive change of your work, where you will make your changes:

```bash
git checkout -b a-descriptive-name-for-my-changes
```
4. Install all dependencies. To do this we recommend using [conda]('https://docs.conda.io/projects/conda/en/latest/index.html) named `scrc` in which you can import the `env.yml` file found in the root directory of this project.
5. Install postgres on your system to host your own database (~26GB). You can then import the backup from the sandbox server (follow steps 2 and 3 of Sandbox setup to connect to the sandbox). A backup of the database is found at `/database/scrc.gz`, however a new one could be made with `sudo -u postgres pg_dump scrc | gzip -9 > scrc.gz`. Locally you can import this database using 
   ```
   scp username@fdn-sandbox3.inf.unibe.ch:/database/scrc.gz ./scrc.gz
   dropdb scrc && createdb scrc
   gunzip < scrc.gz | psql scrc 
   ```
   If you do not want to use the backup but generate the database from scratch follow the steps outlined in [Setup database from scratch](#setup-database-from-scratch)
6. You might want to replicate steps 6 and 7 of the sandbox setup using scp to copy the folders to your local setup.


## Developing

### Before starting

After logging in you should be in the conda `scrc` environment which is visible in your terminal

```bash
(scrc) <USERNAME>@fdn-sandbox3:~$
```

You can find the code in the cloned GitHub directory `SwissCourtRulingCorpus`.
****
### Coding

Make the changes in the files you want to modify or create files accordingly. To get an idea how you might add
something, don't hesitate to explore the code.

If you have edited the code, you can run it via the following command

```bash
python -m PATH.TO.MODULE
```

where the path to the module for example is `scrc.preprocessors.extractors.section_splitter`

If you're happy with your changes add your files and make a commit to record them locally:

```bash
git add <FILE YOU MODIFIED>
git commit
```

For good commit messages use for example the following guide: https://chris.beams.io/posts/git-commit/

It is a good idea to sync your copy of the code with the original repository regularly. This way you can quickly account
for changes:

```bash
git fetch upstream
git rebase upstream/main
```

### Running a pipeline step
Each pipeline step can be run in three different modes.
1. ignore cache flag set to true in the `config.ini` file.
   
   The module will delete and recreate the progress file. This means the step is run again for all spiders.

1. Decision_ids set in the `start` method of the extractors (`clean` method for the `Cleaner`) 

   The module will read the progress file and from the spiders it has not done yet will select only the given decision_ids. It then executes the pipeline for the given decision_ids from the spider not yet in the progress file.
   This can be combined with the ignore cache flag in the `config.ini` to run it for every spider with the given decision_ids 

1. Decision_ids not set in the `start` method of the extractors (`clean` method for the `Cleaner`)

   The module will execute all the decisions of the spider not yet present in the progress file. 
### Opening a Pull Request

Push the changes to your forked repository using:

```bash
git push -u origin a-descriptive-name-for-my-changes
```

Then you can go to your GitHub Repository and open a Pull Request to the main branch of the original repository. Please
make sure to create a good Pull Request by following guidelines such
as [How to Make a Perfect Pull Request](https://betterprogramming.pub/how-to-make-a-perfect-pull-request-3578fb4c112).
Maybe the following list also serves as a good starting
point (https://www.pullrequest.com/blog/writing-a-great-pull-request-description/):

```
## What?
## Why?
## How?
## Testing?
## Screenshots (optional)
## Anything Else?
```

## Tasks

The code relevant for you is in the [scrc/preprocessors](scrc/preprocessors) folder. You don't need to interact with the
PostGres database directly, but the relevant data will be served to you in the respective functions. Which functions you
should pay attention to is described in more detail below.

You are generating ground truth with your tasks. RegEx is probably the easiest way to solve the tasks for a large
portion of the decisions. To increase the recall and precision, you can investigate more sophisticated approaches such
as labelling some outlier decision (not recognized by RegEx) yourself and then training a classifier on it.
Non-functional goals are the following:

- Accuracy is more important than performance since it runs only once
- More spiders implemented are better than ultra-high accuracy since exceptions often tend to be not very useful in
  downstream tasks anyway

### Section Splitting

The Section Splitting task is a prerequisite for the following task and should be performed first if it is not already
done for the spider you are handling. The [section_splitter.py](scrc/preprocessors/extractors/section_splitter.py)
orchestrates the section splitting for all the spiders. If you want to rerun your spider, remove the spider from the
progress file [spiders_section_split.txt](data/progress/spiders_section_split.txt). All spider names present in this
file will be skipped. This works analogously for the other tasks
(files:
[spiders_citation_extracted.txt](data/progress/spiders_citation_extracted.txt),
[spiders_judgement_extracted.txt](data/progress/spiders_judgement_extracted.txt),
[spiders_procedural_participation_extracted.txt](data/progress/spiders_procedural_participation_extracted.txt),
[spiders_court_composition_extracted.txt](data/progress/spiders_court_composition_extracted.txt)).

If your spider has no entry in
the [section_splitting_functions.py](scrc/preprocessors/extractors/spider_specific/section_splitting_functions.py), you
need to add code to split the text into the different sections first.

### Citation Extraction

The [citation_extractor.py](scrc/preprocessors/extractors/citation_extractor.py) is the main file coordinating the
citation extraction for all spiders. When you implement a new spider, you should add your code to
the [citation_extracting_functions.py](scrc/preprocessors/extractors/spider_specific/citation_extracting_functions.py)

### Judgment Outcome Extraction

If you do not get any rulings, please make sure that your spider has an implementation in
the [section_splitting_functions.py](scrc/preprocessors/extractors/spider_specific/section_splitting_functions.py).

The [judgment_extractor.py](scrc/preprocessors/extractors/judgment_extractor.py) is the main file coordinating the
judgment extraction for all spiders. When you implement a new spider, you should add your code to
the [judgement_extracting_functions.py](scrc/preprocessors/extractors/spider_specific/judgment_extracting_functions.py)

### Judicial Person Extraction

If you do not get any header, please make sure that your spider has an implementation in
the [section_splitting_functions.py](scrc/preprocessors/extractors/spider_specific/section_splitting_functions.py).

#### Court Composition

The [court_composition_extractor.py](scrc/preprocessors/extractors/court_composition_extractor.py) is the main file
coordinating the court composition (judges and clerks) extraction for all spiders. When you implement a new spider, you
should add your code to
the [court_composition_extracting_functions.py](scrc/preprocessors/extractors/spider_specific/court_composition_extracting_functions.py)

#### Procedural Participation

The [procedural_participation_extractor.py](scrc/preprocessors/extractors/procedural_participation_extractor.py) is the
main file coordinating the procedural participation (parties and legal counsels) extraction for all spiders. When you
implement a new spider, you should add your code to
the [procedural_participation_extracting_functions.py](scrc/preprocessors/extractors/spider_specific/procedural_participation_extracting_functions.py)

#### Court Chambers

The [court_chambers.csv](legal_info/court_chambers.csv) file encloses information about how the spiders are
broken down into courts. You can improve and simplify your task by dividing the decisions by courts, as you would be
working with a lower amount of decisions within the same court.

When implementing your spider, you can invoke the court chambers from
the [court_chambers.json](legal_info/court_chambers.json) file into your code.

In order to retrieve the courts from the database, you have to make use of the namespace['court'], along with a few if
statements. Whenever we encounter a certain court, we would like to execute these particular regular expressions and
when we encounter a different court to take these alternative regular expressions and so on so forth. You can view an
example of court chambers being applied in a section splitting task for
this [specific spider](https://github.com/JoelNiklaus/SwissCourtRulingCorpus/blob/3a177d02cdd87eba07aa4d3bca4e2fb52995cb18/scrc/preprocessors/extractors/spider_specific/section_splitting_functions.py#L112-L190)
.
## Running long tasks

Some tasks may take a very long time to run. This means if you just start it in the terminal, the terminal will
disconnect and thus abort your task. To prevent this, you can use a terminal tool called tmux on the sandbox. To find
out how it works, just google "tmux tutorial". You will find great tutorials, like the
following: https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/
, https://linuxize.com/post/getting-started-with-tmux/, https://leimao.github.io/blog/Tmux-Tutorial/.

## SQL FAQ

An image with the current database table setup is stored under [database_creation](./database_creation/DB_Schema.png).
This section explains the methods of the sql_select_utils so they can be replicated as SQL Queries. If you want to check for what a specific case return sand it is not explained in here, then you can retrieve the result easily:
1. Change to the default project path and open a python interpreter
   ```bash
   $   python3
   ```
2. import the needed function
   ```python
      from scrc.utils.sql_select_utils import map_join
   ```
3. call the function with your arguments
   ```python
      map_join('text', 'file_numbers', 'file_number')
   ```
4. You should then get the output into your terminal
### Joining Tables on queries with decision table
Simplest query is `Select * FROM decision`, however it can be made more complex. Instead of querying all the fields (`*`, some fields can be specified e.g. decision.decision_id, it is needed to specify the table if the "FROM" table and the "JOIN" table have the same key.)
#### Language
```<Query> LEFT JOIN language ON language.language_id = decision.language_id```
to make this addon on the query `join_decision_on_language()` can be used. 

#### File
A similar function is implemented with `join_file_on_decision()`, which is the same as specifying an SQL query ```<Query> LEFT JOIN file ON file.file_id = decision.file_id```

### Joining the decision table on other tables
For joining the decision table on to a specified table and field combination the `join_decision_on_parameter` function can be used, which will result in ```<Query> LEFT JOIN decision on decision.<FIELDNAME> = <TABLE>.<FIELDNAME> ```

If the language has to be joined aswell then the function `join_decision_and_language_on_parameter` can be used which will return the same result but with the language table joined aswell (```<Query> LEFT JOIN decision on decision.<FIELDNAME> = <TABLE>.<FIELDNAME> LEFT JOIN language ON language.language_id = decision.language_id```)

### Where clause helper

#### Checking for certain spiders
If only the decisions of certain spider should be selected, then the following can be used:
Use by <TABLE>.<FIELDNAME> IN where_string_spider(<FIELDNAME>, <SPIDER>)
This is the same as writing in SQL: ```<Query> FROM <TABLE>.<FIELDNAME> IN (SELECT <FIELDNAME> from decision WHERE chamber_id IN (SELECT chamber_id FROM chamber WHERE spider_id IN (SELECT spider_id FROM spider WHERE spider.name = '<SPIDERNAME>')))```


### Saving to database
This is a routine in the sql_select_utils which cannot be translated into one single SQL query. For the inserts the pandas to_sql method is used. Inserting decisions by hand should not be necessary normally.
Some queries used in the routine are:

#### Selecting file entries by file_name

```SELECT file_id FROM file WHERE file_name = '<FILENAME>'```
#### Selecting file entries by file_name
A simple selct statement
```SELECT chamber_id FROM chamber WHERE chamber_string = '<CHAMBERNAME>'```

and more very simple select statements

#### Selecting decisions to delete
If certain decisions have to be deleted the following where clause is used internally:
``` WHERE decision_id IN (1,2,3) ``` where 1,2,3 are the decision_ids of the decisions to be deleted.

### JOIN and map_join

#### join
Can be used to join a table on to another table. 
```LEFT JOIN <TABLE> ON <TABLE>.<FIELDNAME> = <JOIN_TABLE>.<FIELDNAME>```.
In the sql_select_utils file the default case is joining to decision_id on table decision


#### map_join: Simple
To aggregate different rows from a table into a different field on query the map_join can be used, which returns the following 'JOIN' string:
```LEFT JOIN (SELECT <TABLE>.<FIELDNAME_TO_BE_GROUPED>, array_agg(<TABLE>.<FIELDNAME_TO_BE_MAPPED>) <NEW_NAME_FOR_AGG_FIELD> <ADDITIONAL_FIELDS_TO_EXTRACT> FROM <TABLE> GROUP BY <FIELDNAME_TO_BE_GROUPED>) as <TABLE> ON <TABLE>.<FIELDNAME_TO_BE_GROUPED> = <TABLE_TO_BE_JOINED_ON>.<FIELDNAME_TO_BE_GROUPED>```. Fieldname_to_be_grouped specifies which field the rows share, which for example could be decision_id.

An example of such a query is using `map_join('text', 'file_numbers', 'file_number')` which results in  ` LEFT JOIN (SELECT file_number.decision_id, array_agg(file_number.text) file_numbers  FROM file_number GROUP BY decision_id) as file_number ON file_number.decision_id = d.decision_id'`

#### map_join: With filling
This can be used the same as the simple version, however in the table to be aggregated some fields can be replaced e.g. ids can be replaced by another field in another table. 
A usage example: 
`map_join('judgment_id', 'judgments', 'judgment_map', fill={'table_name': 'judgment', 'field_name': 'text', 'join_field': 'judgment_id'})` returns ```LEFT JOIN (SELECT judgment_map_mapped.decision_id, json_strip_nulls(json_agg(json_build_object('text', text))) judgments FROM (SELECT text, judgment_map.decision_id FROM judgment_map LEFT JOIN judgment  ON judgment.judgment_id = judgment_map.judgment_id) as judgment_map_mapped GROUP BY decision_id) as judgment_map ON judgment_map.decision_id = d.decision_id```
which is the judgment_map added to the decisions with the judmgent_ids replaced by the text saved in judgment, that means the judmgents column is added tn the query which contains an array of all judgments a decision has, served in text form.

### Selecting "everything"
The usage of the `select_paragraphs_with_decision_and_meta_data` function is to `SELECT <RETURN_TUPLE_SECOND_ENTRY> FROM <RETURN_TUPLE_FIRST_ENTRY> WHERE <WHATEVER_YOU_WANT>
#### tables
`select_paragraphs_with_decision_and_meta_data` is used to join multiple tables. The first entry in the tuple returns the table selection string:
```decision d LEFT JOIN file ON file.file_id = d.file_id  LEFT JOIN (SELECT section_mapped.decision_id, json_strip_nulls(json_agg(json_build_object('name', name,'section_text', section_text))) sections FROM (SELECT name, section_text, section.decision_id FROM section LEFT JOIN section_type  ON section_type.section_type_id = section.section_type_id) as section_mapped GROUP BY decision_id) as section ON section.decision_id = d.decision_id  LEFT JOIN lower_court ON lower_court.decision_id = d.decision_id  LEFT JOIN (SELECT citation_mapped.decision_id, json_strip_nulls(json_agg(json_build_object('name', name,'text', text,'url', url))) citations FROM (SELECT name, text, url, citation.decision_id FROM citation LEFT JOIN citation_type  ON citation_type.citation_type_id = citation.citation_type_id) as citation_mapped GROUP BY decision_id) as citation ON citation.decision_id = d.decision_id  LEFT JOIN (SELECT judgment_map_mapped.decision_id, json_strip_nulls(json_agg(json_build_object('text', text))) judgments FROM (SELECT text, judgment_map.decision_id FROM judgment_map LEFT JOIN judgment  ON judgment.judgment_id = judgment_map.judgment_id) as judgment_map_mapped GROUP BY decision_id) as judgment_map ON judgment_map.decision_id = d.decision_id  LEFT JOIN (SELECT file_number.decision_id, array_agg(file_number.text) file_numbers  FROM file_number GROUP BY decision_id) as file_number ON file_number.decision_id = d.decision_id LEFT JOIN (SELECT paragraph_mapped.decision_id, json_strip_nulls(json_agg(json_build_object('paragraph_text', paragraph_text,'section_type_id', section_type_id,'section_id', section_id))) paragraphs FROM (SELECT paragraph_text, section_type_id, paragraph.section_id, paragraph.decision_id FROM paragraph LEFT JOIN section  ON section.section_id = paragraph.section_id) as paragraph_mapped GROUP BY decision_id) as paragraph ON paragraph.decision_id = d.decision_id```

The tables added are:
- judgment
- citation
- file
- section
- paragraph
- file_number
- lower_court

#### fields
The second entry in the table returns the fields
```d.*, extract(year from d.date) as year, judgments, citations, file.file_name, file.html_url, file.pdf_url, file.html_raw, file.pdf_raw, sections, paragraphs, file_numbers, lower_court.date as origin_date, lower_court.court_id as origin_court, lower_court.canton_id as origin_canton, lower_court.chamber_id as origin_chamber, lower_court.file_number as origin_file_number```

#### Setup database from scratch
Use the file `drop_and_create_tables.sql` to drop the tables and set them up again, then use the `setup_values.sql` to insert basic values into the freshly generated tables again. If you want to have a more up-to-date version of the setup values, then you have to create the table code which should be more up-to-date with the setup_values_creation.py.

## Common Errors
### PermissionError: [Errno 13] Permission denied: '/tmp/tika.log'
Remove the tika.log file in `/tmp` using `sudo rm /tmp/tika.log`.

### idle_in_transaction_session_timeout
This is a postgres error that can be resolved by updating the configuration, such that the idle_in_transaction_session_timout is disabled. To do this open the postgres console and execute `SET idle_in_transaction_session_timeout TO '0'`, which should be the default state.
## Questions?

Do not hesitate to contact Adrian Joerg or Joel Niklaus via email: {firstname}.{lastname}@inf.unibe.ch
