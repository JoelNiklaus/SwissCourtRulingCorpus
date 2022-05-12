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

### Local setup

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
5. Install postgres on your system to host your own database (~26GB). You can then import the backup from the sandbox server (follow steps 2 and 3 of Sandbox setup). A backup of the database is found at `/database/scrc.gz`, however a new one could be made with `sudo -u postgres pg_dump scrc | gzip -9 > scrc.gz`. Locally you can import this database using 
   ```
   scp username@fdn-sandbox3.inf.unibe.ch:/database/scrc.gz ./scrc.gz
   dropdb scrc && createdb scrc
   gunzip < scrc.gz | psql scrc 
   ```
6. You might want to replicate steps 6 and 7 using a scp to copy the folders to your local setup.


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

## Questions?

Do not hesitate to contact Adrian Joerg or Joel Niklaus via email: {firstname}.{lastname}@inf.unibe.ch
