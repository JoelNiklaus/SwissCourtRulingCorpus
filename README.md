# SwissCourtRulingCorpus

## Overview of process done in building this dataset

For an overview of the process used to construct this dataset, please consult the file [scrc/main.py](scrc/main.py).

For extracting the raw text content from pdf files we used tika and for html files we used BeautifulSoup.

The Swiss Court Ruling Corpus (SCRC) contains Swiss court rulings. It was scraped from http://entscheidsuche.ch/docs on
February 1st 2021.

More information about this dataset can be found in the following
spreadsheet: https://docs.google.com/spreadsheets/d/1bzW_7kpRAYxSis15g3s0HX8OcyXfWGgfJfybII9fN5s/edit?usp=sharing

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
ssh -N -f -L localhost:8888:localhost:8888 <your-username>@fdn-sandbox3.inf.unibe.ch
```

## Postgres

The data is stored in a Postgres DB so we can avoid loading into RAM the total data in order to work with it.

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

## useful DB queries

Amount of approvals for a specific court:
```bash
SELECT count(*) FROM judgment
            LEFT JOIN judgment_map ON judgment_map.judgment_id = judgment.judgment_id
            LEFT JOIN decision ON decision.decision_id = judgment_map.decision_id
            LEFT JOIN chamber ON chamber.chamber_id = decision.chamber_id
            LEFT JOIN spider ON spider.spider_id = chamber.spider_id
            WHERE spider.name = 'BS_Omni'
            AND judgment.judgment_id = 1
```
### The following two queries are integrated within the judgment_pattern_extractor module. Check out the modules readme to execute them automatically


Amount of total judgment outcomes for a specific court (decisions with 2 or more judgments counted once):
```bash
SELECT count(DISTINCT d.decision_id) FROM decision d 
    LEFT JOIN chamber c ON c.chamber_id = d.chamber_id 
    LEFT JOIN spider sp ON sp.spider_id = c.spider_id 
    LEFT JOIN judgment_map j ON j.decision_id = d.decision_id 
    WHERE judgment_id IS NOT NULL 
    AND sp.name = 'CH_BGer' 
```

Amount of sections of section_type 6 (rulings) found for a specific court (change last line's number to desired section_type):
```bash
SELECT count(*) FROM decision d
    LEFT JOIN section s ON d.decision_id = s.decision_id
    LEFT JOIN section_type t ON t.section_type_id = s.section_type_id
    LEFT JOIN chamber c ON c.chamber_id = d.chamber_id
    LEFT JOIN spider sp ON sp.spider_id = c.spider_id
    WHERE sp.name = 'CH_BGer'
    AND section_text != ''
    AND s.section_type_id = 6
        ```
