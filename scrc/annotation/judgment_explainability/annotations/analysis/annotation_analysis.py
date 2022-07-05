"""
- Cohen’s Kappa, Fleiss’s, Krippendorff’s Alpha
- ROUGE-L, ROUGE-1, ROUGE-2 (Lin, 2004)
- BLEU (Papineni et al., 2001) (unigram and bigram averaging)
- METEOR (Lavie and Agarwal, 2007)
- Jaccard Similarity
- Overlap Maximum and Overlap Minimum
- BARTScore (Yuan et al., 2021) and BERTScore Zhang et al. (2020).
- By looking at the annotated sentences themselves and at the reasoning in the free-text annotation for some of the more complex cases4 a qualitative analysis of
the annotation is also possible.

Sources for Metrics:
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html#sklearn.metrics.cohen_kappa_score
- https://www.statsmodels.org/stable/_modules/statsmodels/stats/inter_rater.html
- https://pypi.org/project/rouge-score/
- https://www.journaldev.com/46659/bleu-score-in-python
- https://stackoverflow.com/questions/63778133/how-can-i-implement-meteor-score-when-evaluating-a-model-when-using-the-meteor-s
- https://pypi.org/project/bert-score/
- https://github.com/neulab/BARTScore
- https://pyshark.com/jaccard-similarity-and-jaccard-distance-in-python/





"""
import json
from pathlib import Path

import pandas
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import collections
from random import randint
import nltk
nltk.download('punkt')

LANGUAGES = ["de", "fr", "it"]
PERSONS = ["angela", "lynn", "thomas"]
LABELS = [ "Lower court","Opposes judgment", "Supports judgment"]
CONTROL_SPANS_COUNT = {}
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def remove_dublicate(list_dublicate: list) -> list:
    return list(dict.fromkeys(list_dublicate))


def to_csv(filepath: Path, df: pd.DataFrame):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)


def dump_csv(datasets):
    for data in datasets:
        to_csv(Path(datasets[data].index.name + "_complete.csv"), datasets[data])
        try:
            user_input = datasets[data][datasets[data]["user_input"] != ""]
            to_csv(Path(datasets[data].index.name + "_user_input.csv"),
                   user_input[['id_scrc', '_annotator_id', "user_input"]])
            case_not_accepted = datasets[data][datasets[data]["answer"] != "accept"]
            to_csv(Path(datasets[data].index.name + "_ig_re.csv"),
                   case_not_accepted[['id_scrc', '_annotator_id', "user_input"]])
        except KeyError:
            print(datasets[data].index.name.capitalize() + " does not have a user input!")


def extract_dataset() -> dict:
    datasets = {}
    for language in LANGUAGES:
        a_list = []
        with open("../test.jsonl".format(language), "r") as json_file:
        #with open("../annotations_{}.jsonl".format(language), "r") as json_file:
            json_list = list(json_file)
            for json_str in json_list:
                result = json.loads(json_str)
                a_list.append(result)
                assert isinstance(result, dict)
                dfItem = pd.DataFrame.from_records(a_list)
                dfItem = dfItem.set_index("id_scrc")
                dfItem.index.name = "annotations_{}".format(language)
                datasets["annotations_{}".format(language)] = dfItem
        for person in PERSONS:
            a_list = []
            try:
                with open("../annotations_{}-{}.jsonl".format(language, person), "r") as json_file:
                    json_list = list(json_file)
                    for json_str in json_list:
                        result = json.loads(json_str)
                        a_list.append(result)
                        assert isinstance(result, dict)
                        dfItem = pd.DataFrame.from_records(a_list)
                        dfItem.index.name = "annotations_{}-{}".format(language, person)
                        datasets["annotations_{}-{}".format(language, person)] = dfItem
            except FileNotFoundError:
                pass
    return datasets


def extract_values_from_column(annotations: pandas.DataFrame, col_1:str, col_2:str) -> pandas.DataFrame:
    annotations_col = annotations.explode(col_1).reset_index().drop([col_2], axis=1)
    df_col = annotations_col[col_1].apply(pd.Series).add_prefix("{}_".format(col_1))
    annotations_col = annotations_col.drop([col_1], axis=1)
    return annotations_col.join(df_col)

def get_span_df(annotations_spans:pandas.DataFrame, annotations_tokens:pandas.DataFrame,  span:str,) -> (pandas.DataFrame, dict):
    spans = annotations_spans[annotations_spans["spans_label"] == span]
    token_numbers = {}
    for mini_list in list(spans[['annotations_de','_annotator_id','spans_token_start', 'spans_token_end']].values):
        numbers = []
        for nr in list(range(int(mini_list[2]), int(mini_list[3])+1)):
            numbers.append(nr)
        token_numbers["{}.{}.{}".format(mini_list[0],mini_list[1],randint(0, 100000))] = numbers
    spans_list = []
    for key in token_numbers:
        new_annotations_tokens = annotations_tokens[annotations_tokens['annotations_de'] == int(key.split(".")[0])].copy()
        new_annotations_tokens = new_annotations_tokens[new_annotations_tokens["tokens_id"].isin(token_numbers[key])]
        new_annotations_tokens = new_annotations_tokens[new_annotations_tokens['_annotator_id'] == key.split(".")[1]]
        spans_list.append(new_annotations_tokens)
    spans = pd.concat(spans_list)
    return spans

def get_annotator_set(annotator_name: str, tokens: pandas.DataFrame)-> pandas.DataFrame:
    annotator = tokens[tokens['_annotator_id'] == "annotations_de-{}".format(annotator_name)].drop_duplicates().copy()
    annotator['tokens_text'] = annotator.groupby(['annotations_de'])['tokens_text'].transform(lambda x: ' '.join(x))
    annotator['tokens_id'] = annotator.groupby(['annotations_de'])['tokens_id'].transform(lambda x: ','.join(x.astype(str)))
    #.transform(lambda x: [x.tolist()]*len(x))z
    annotator = annotator[['annotations_de','tokens_text','tokens_id','_timestamp']]
    annotator= annotator.drop_duplicates()
    annotator["tokens_id"] = annotator["tokens_id"].astype(str).str.split(",")
    return annotator

def string_to_int_list(column_value) -> list:
    return_list = []
    if type(column_value) is str:
        for integer in column_value.split(","):
            return_list.append(int(integer))
        return return_list
    if type(column_value) is int:
        return [column_value]


if __name__ == '__main__':
    datasets = extract_dataset()
    dump_csv(datasets)
    number_of_spans = {"Lower court": 0,"Opposes judgment":0, "Supports judgment":0}
    for lang in LANGUAGES:
        try:
            #print(collections.Counter(list(datasets["annotations_{}".format(lang)].index)))
            annotations = datasets["annotations_{}".format(lang)][
                datasets["annotations_{}".format(lang)]["answer"] == "accept"]
            annotations_spans = extract_values_from_column(annotations, "spans", "tokens")
            annotations_tokens = extract_values_from_column(annotations, "tokens", "spans")
            for label in LABELS:
                label_df = get_span_df(annotations_spans, annotations_tokens, label)
                for pers in PERSONS:
                    globals()[f"{label.lower().replace(' ','_')}_{pers}"] = get_annotator_set(pers, label_df)
        except KeyError:
            pass

    #print(lower_court_thomas)
    #print(lower_court_angela)
    #print(lower_court_lynn)

    merged_df = pd.merge(lower_court_thomas,lower_court_angela , on="annotations_de", suffixes=('_thomas', '_angela'), how ="left").fillna("Nan")
    df = pd.merge(merged_df,lower_court_lynn , on="annotations_de", how ="right").fillna("Nan").rename(columns={"tokens_text":"tokens_text_lynn","tokens_id":"tokens_id_lynn"})




    normalized_tokens = []
    df_copy = df.copy()
    df["tokens_text"] = df_copy["tokens_text_thomas"].astype(str)+ " " + df_copy["tokens_text_angela"].astype(str) + " " + df_copy["tokens_text_lynn"].astype(str)

    for sentence in df["tokens_text"].values:
        tokens = nltk.word_tokenize(sentence, language='german')
        normalized_tokens.append(dict(zip(tokens,range(0, len(tokens)))))
    df["normalized_tokens"] = normalized_tokens

    normalized_tokens = []
    for sentences in df[["annotations_de","tokens_text_thomas"]].values:
        normalized_tokens_row = []
        tokens = nltk.word_tokenize(sentences[1], language='german')
        token_dict = df[df["annotations_de"] == sentences[0]]["normalized_tokens"].values[0]
        for word in tokens:
            normalized_tokens_row.append(token_dict[word])
        normalized_tokens.append(normalized_tokens_row)

    df["normalized_tokens_thomas"] = normalized_tokens

    print(df[["normalized_tokens","tokens_id_thomas","normalized_tokens_thomas"]])



    #cohen_kappa_score = cohen_kappa_score(df["tokens_id_thomas"], df["tokens_id_angela"])
    #print(cohen_kappa_score)

    #df["cohen_kappa_score"] = cohen_kappa_score(df["tokens_id_thomas"], df["tokens_id_angela"])



