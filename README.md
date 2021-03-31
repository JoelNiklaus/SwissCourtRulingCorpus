# SwissCourtRulingCorpus

## Overview of process done in building this dataset

1. Scrape only the source files and store them in a folder named after the dataset_path (scraper.py)
2. For each json file (metadata) look for a html and a pdf file with the same name.Extract the content and store it raw
   in a pandas dataframe together with the filename and any metadata in additional columns. Store the files in a folder
   csv/raw (extractor.py)
3. Clean the raw content and store the resulting dataframe in a different folder csv/clean (cleaner.py)
4. Aggregate all the courts into one file _all.csv and into language separated files _de.csv, _fr.csv and _it.csv (
   aggregator.py)

For extracting the raw text content from pdf files we used tika and for html files we used BeautifulSoup.

The Swiss Court Ruling Corpus (SCRC) contains Swiss court rulings. It was scraped from http://entscheidsuche.ch/docs on
February 1st 2021.

More information about this dataset can be found in the following
spreadsheet: https://docs.google.com/spreadsheets/d/1bzW_7kpRAYxSis15g3s0HX8OcyXfWGgfJfybII9fN5s/edit?usp=sharing

Time: 2010 or 2015 until 2020

Task: Named Entity Recognition and Relation Extraction

Export conda env with:
```conda env export > env.yml --no-builds```

## Info for Annotators

Go to http://fdn-sandbox3.inf.unibe.ch:8080/?session={firstname}
(this session firstname can then be used to compare the different annotators and compute agreement scores)

## Design Choices

We went for fasttext for language identification because of this
benchmark: https://towardsdatascience.com/benchmarking-language-detection-for-nlp-8250ea8b67c

## Notebooks

Follow the instructions on https://ljvmiranda921.github.io/notebook/2018/01/31/running-a-jupyter-notebook/ to run a
jupyter notebook from the server locally.

Run the following commands to enable tabnine autocompletion also in jupyter notebook

```bash
python -m pip install jupyter-tabnine
jupyter nbextension install --py jupyter_tabnine
jupyter nbextension enable --py jupyter_tabnine
jupyter serverextension enable --py jupyter_tabnine
```

### Start notebook server (run on server)

```bash
jupyter notebook --no-browser --port=8888 --notebook-dir=scrc/notebooks
```

### Forward port to local machine (run on local machine)

```bash
ssh -N -f -L localhost:8888:localhost:8888 fdn-admin@fdn-sandbox3.inf.unibe.ch
```

## Pandas Memory usage

When pandas loads from a csv file and creates a dataframe in memory, this dataframe will allocate more than 2x the size
of the csv! Make sure you load big csv files in chunks!

## Handy bash commands

Print files sorted by size

```bash
du -h data/csv/clean/* | sort -h
```

## Troubleshooting

Sometimes ``pip install`` says that the requirement is already satisfied. If the module is not found
try ``python -m pip install ...``.

Dask does not work with csv files containing new lines inside fields because it cannot process csv chunks