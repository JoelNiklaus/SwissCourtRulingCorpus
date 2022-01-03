# Contributors

## Setup

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
ln -s /home/fdn-admin/SwissCourtRulingCorpus/data/ ~/SwissCourtRulingCorpus/data/spiders
```

## Developing

### Before starting

After logging in you should be in the conda `scrc` environment which is visible in your terminal

```bash
(scrc) <USERNAME>@fdn-sandbox3:~$
```

You can find the code in the cloned GitHub directory `SwissCourtRulingCorpus`.

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

### Opening a Pull Request

Push the changes to your forked repository using:

```bash
git push -u origin a-descriptive-name-for-my-changes
```

Then you can go to your GitHub Repository and open a Pull Request to the main branch of the original repository.
Please make sure to create a good Pull Request by following guidelines such as [How to Make a Perfect Pull Request](https://betterprogramming.pub/how-to-make-a-perfect-pull-request-3578fb4c112). Maybe the following list also serves as a good starting point (https://www.pullrequest.com/blog/writing-a-great-pull-request-description/):
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
coordinating the court composition (judges and clerks) extraction for all spiders. When you implement a new spider, you should add your code to
the [court_composition_extracting_functions.py](scrc/preprocessors/extractors/spider_specific/court_composition_extracting_functions.py)

#### Procedural Participation

The [procedural_participation_extractor.py](scrc/preprocessors/extractors/procedural_participation_extractor.py) is the
main file coordinating the procedural participation (parties and legal counsels) extraction for all spiders. When you implement a new spider, you should add your
code to
the [procedural_participation_extracting_functions.py](scrc/preprocessors/extractors/spider_specific/procedural_participation_extracting_functions.py)

## Questions?

Do not hesitate to contact Adrian Joerg or Joel Niklaus via email: {firstname}.{lastname}@inf.unibe.ch
