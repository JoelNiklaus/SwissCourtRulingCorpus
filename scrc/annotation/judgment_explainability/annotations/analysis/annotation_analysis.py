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
- https://scikit-learn.org/stable/modules/model_evaluation.html#cohen-kappa
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
        #with open("../test.jsonl".format(language), "r") as json_file:
        with open("../annotations_{}.jsonl".format(language), "r") as json_file:
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
    annotator = annotator[['annotations_de','tokens_text','tokens_id']]
    annotator= annotator.drop_duplicates()
    annotator["tokens_id"] = annotator["tokens_id"].astype(str).str.split(",")
    return annotator

"""
@Todo Create merge loop for even and odd lists
"""
def merge_label_df(df_list: list, person_suffixes: list):
    i = 0
    df_list = sorted(df_list, key=len)
    merged_df = pd.merge(df_list[i], df_list[i + 1], on="annotations_de",
                         suffixes=('_{}'.format(person_suffixes[i]), '_{}'.format(person_suffixes[i + 1])),
                         how="left").fillna("Nan")
    df = pd.merge(merged_df, df_list[i + 2], on="annotations_de", how="right").fillna("Nan").rename(
        columns={"tokens_text": "tokens_text_{}".format(person_suffixes[i + 2]),
                 "tokens_id": "tokens_id_{}".format(person_suffixes[i + 2])})
    return df

def get_normalize_tokens_dict(df: pandas.DataFrame) -> pandas.DataFrame:
    normalized_tokens = []
    df_copy = df.copy()
    df["tokens_text"] = df_copy["tokens_text_thomas"].astype(str) + " " + df_copy["tokens_text_angela"].astype(
        str) + " " + df_copy["tokens_text_lynn"].astype(str)
    for sentence in df["tokens_text"].values:
        tokens = nltk.word_tokenize(sentence, language='german')
        normalized_tokens.append(dict(zip(tokens, range(0, len(tokens)))))
    df["normalized_tokens_dict"] = normalized_tokens
    return df

def normalize_tokens(df: pandas.DataFrame,pers: str, lang: str):
    normalized_tokens = []
    for sentences in df[["annotations_de","tokens_text_{}".format(pers)]].values:
        normalized_tokens_row = []
        tokens = nltk.word_tokenize(sentences[1], language=lang)
        token_dict = df[df["annotations_de"] == sentences[0]]["normalized_tokens_dict"].values[0]
        for word in tokens:
            normalized_tokens_row.append(token_dict[word])
        normalized_tokens.append(normalized_tokens_row)
    df["normalized_tokens_{}".format(pers)] = normalized_tokens
    return df

def normalize_list_length(list_1: list, list_2: list) -> (list, list):
    while len(list_1) != len(list_2):
        if len(list_1) > len(list_2):
            random = randint(0, 1000)
            if random not in list_2 and random not in list_1:
                list_2.append(random)
        if len(list_1) < len(list_2):
            random = randint(0, 1000)
            if random not in list_2 and random not in list_1:
                list_1.append(random)
    return list_1, list_2


if __name__ == '__main__':
    datasets = extract_dataset()
    dump_csv(datasets)
    label_list = []
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
                    label_list.append(get_annotator_set(pers, label_df))
                globals()[f"{label.lower().replace(' ', '_')}"] = merge_label_df(label_list,PERSONS)
        except KeyError:
            pass

    lower_court = get_normalize_tokens_dict(lower_court)
    for pers in PERSONS:
        lower_court = normalize_tokens(lower_court, pers, "german")

    lower_court.drop('tokens_text', axis=1,inplace=True)
    lower_court = lower_court.loc[lower_court.astype(str).drop_duplicates().index]

    print(lower_court[lower_court["annotations_de"]==436297])
    for value_list in lower_court[['annotations_de',"normalized_tokens_angela","normalized_tokens_lynn", "normalized_tokens_thomas"]].values:
        for i in range (1, len(value_list)-1, 2):
            c_k_1, c_k_2, c_k_3 = 2,2,2
            list_a, list_b, list_c = value_list[i], value_list[i+1], value_list[i+2]
            if len(list_a) != 1 and len(list_b) != 1:
                list_a, list_b = normalize_list_length(value_list[i], value_list[i + 1])
                c_k_1 = cohen_kappa_score(list_a, list_b )
            if len(list_a) != 1 and len(list_c) != 1:
                list_a, list_c = normalize_list_length(value_list[i], value_list[i + 2])
                c_k_2 = cohen_kappa_score(list_a, list_c )
            if len(list_b) != 1 and len(list_c) != 1:
                list_b, list_c = normalize_list_length(value_list[i + 1], value_list[i + 2])
                c_k_3 = cohen_kappa_score(list_b, list_c )

            c_k_list = [c_k_1,c_k_2,c_k_3]
            try:
                c_k_list = list(filter((2).__ne__,  c_k_list))
            except ValueError:
                pass

            if len(c_k_list) > 0:
                max_value = max(c_k_list)
                min_value = min(c_k_list)
                avg_value = 0 if len(c_k_list) == 0 else sum(c_k_list)/len(c_k_list)

                print("{},{},{},{},{}".format(value_list[0],len(c_k_list),max_value,min_value,avg_value))









