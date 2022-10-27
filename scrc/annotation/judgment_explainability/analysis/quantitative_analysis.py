import ast
import csv
import os
from pathlib import Path

import preprocessing
import pandas as pd

from scrc.annotation.judgment_explainability.analysis.preprocessing import extract_dataset

LANGUAGES = ast.literal_eval(os.getenv("LANGUAGES"))
LABELS = ast.literal_eval(os.getenv("LABELS"))
PERSONS = ast.literal_eval(os.getenv("PERSONS"))

AGGREGATIONS = preprocessing.AGGREGATIONS
GOLD_SESSION = {"de":"gold_final", "fr":"gold_nina", "it":"gold_nina"}
LEGAL_AREAS = ["penal_law", "social_law", "civil_law"]
YEARS = ["2015", "2016", "2017", "2018", "2019", "2020"]
DF_LIST = []

"""
Bei De wurden noch Fälle hinzugefügt und versucht am ende ein möglichst balnciertes datenset zu haben, bei gold wurde dann aber noch
ein neuer fall rejected was diese zahlen wieder spiegeln,
it daselbe aber hier wurden nie hinzugefügt da diese und französische annotationen nur der vervollständigung und der ground truth dienen
es müssen in zukunft noch mehr annotationen gemacht werden
"""
def count_not_accept(dataset: pd.DataFrame):
    return len(dataset[dataset["answer"] != "accept"].groupby("id_csv"))

def count_user_input(dataset: pd.DataFrame):
    return len(dataset[dataset["user_input"].notnull()].groupby("id_csv"))

"""
Hier betonen das es um akzeptierte fälle von gold standard geht
"""
def count_total_cases(dataset: pd.DataFrame):
    return len(dataset[dataset["answer"] == "accept"])

def count_is_correct(dataset: pd.DataFrame):
    annotations = dataset[dataset["answer"] == "accept"]
    return len(annotations[annotations["is_correct"]==True].groupby("id_csv"))
"""
Hier betonen das es um akzeptierte fälle von gold standard geht
"""

def apply_aggregation(annotations: pd.DataFrame, col_name: str, aggregation:str):
    if aggregation== "mean":
        return annotations.groupby(["year", "legal_area"])[col_name].mean().to_frame().reset_index()
    if aggregation == "max":
        return annotations.groupby(["year", "legal_area"])[col_name].max().to_frame().reset_index()
    if aggregation == "min":
        return annotations.groupby(["year", "legal_area"])[col_name].min().to_frame().reset_index()

def get_agg_table(annotations: pd.DataFrame, col_name: str):
    df = pd.DataFrame(index=YEARS, columns=LEGAL_AREAS)
    df.index.name = "year"
    for index, row in annotations.iterrows():
        ser = pd.Series({row["legal_area"]: row[col_name]})
        ser.name = row["year"]
        df = df.append(ser)
    annotations= pd.DataFrame()
    for la in LEGAL_AREAS:
        annotations[la] = df[la].dropna()

    return annotations

def get_total_agg(annotations: pd.DataFrame,aggregation:str, col_name: str):
    if aggregation == "mean":
        la = annotations.groupby(["legal_area"])[col_name].mean()
        year = annotations.groupby(["year"])[col_name].mean()
    if aggregation == "max":
        la = annotations.groupby(["legal_area"])[col_name].max()
        year = annotations.groupby(["year"])[col_name].max()
    if aggregation == "min":
        la = annotations.groupby(["legal_area"])[col_name].min()
        year = annotations.groupby(["year"])[col_name].min()
    la.name = f"{aggregation}_legal_area"
    return la,year

def count_fact_length(dataset: pd.DataFrame):
    annotations = dataset[dataset["answer"] == "accept"]
    annotations["text"] = annotations["text"].str.len()
    annotations["tokens"] = annotations["tokens"].map(len)
    for agg in AGGREGATIONS:
        char = get_agg_table(apply_aggregation(annotations, "text", agg), "text")
        char = char.append(get_total_agg(annotations, agg,"text")[0])
        char[f"{agg}_year"] =get_total_agg(annotations, agg, "text")[1]
        char.index.name = f"char_{agg}"
        tokens = get_agg_table(apply_aggregation(annotations, "tokens", agg), "tokens")
        tokens = tokens.append(get_total_agg(annotations, agg, "tokens")[0])
        tokens[f"{agg}_year"] = get_total_agg(annotations, agg, "tokens")[1]
        tokens.index.name = f"tokens_{agg}"
        DF_LIST.append(char)
        DF_LIST.append(tokens)

"""
Hier darauf eingehen, dass das deutsche Datenset am unausgeglichesten ist weil am meisten daran gearbeitet wurde.
Man hat mit balanciertem set gestarten nach annotationen etwas weniger balanciert, wollte nicht manuell fälle herauswerfen.
gold standrad noch abeglenht -> time constraint, proof of concept für zukünftige arbeiten ein ziel die ausgeglichenheit zu verfeinern



"""

def get_length(string:str):
    if string != "Nan":
        return len(ast.literal_eval(string))
    else:
        return 0


def analysis():
    extracted_datasets_gold = extract_dataset("../legal_expert_annotations/{}/gold/gold_annotations_{}.jsonl",
                                                  "../legal_expert_annotations/{}/gold/gold_annotations_{}-{}.jsonl")
    extracted_datasets_3 = extract_dataset("../legal_expert_annotations/{}/annotations_{}_merged.jsonl",
                                               "../legal_expert_annotations/{}/annotations_{}_merged-{}.jsonl")

    for l in LANGUAGES:
        for label in LABELS:
            label_df = preprocessing.read_csv(f"{l}/{label.lower().replace(' ', '_')}_{l}_3.csv", "index")
            for person in PERSONS:
                try:
                    label_df[f"length_{person}"] = label_df[f"tokens_id_{person}"].apply(lambda x: get_length(x))
                    numbers = pd.DataFrame({"total_number_of_cases": [
                        count_total_cases(extracted_datasets_gold[f"annotations_{l}-{GOLD_SESSION[l]}"])],
                                            "number_of_comments": [
                                                count_user_input(extracted_datasets_3[f"annotations_{l}"])],
                                            "number_of_not_accepted": [count_not_accept(
                                                extracted_datasets_gold[f"annotations_{l}-{GOLD_SESSION[l]}"])],
                                            "number_of_is_correct": [count_is_correct(
                                                extracted_datasets_gold[f"annotations_{l}-{GOLD_SESSION[l]}"])]})

                    count_fact_length(extracted_datasets_gold[f"annotations_{l}-{GOLD_SESSION[l]}"])

                    with open(Path(f"{l}/quantitative/quantitative_analysis_{l}.csv"), "w") as f:
                        f.truncate()
                        numbers.to_csv(f)
                        f.write("\n")
                        f.write("{},{},{}".format(label, person, label_df[f"length_{person}"].mean()))
                        for df in DF_LIST:
                            df.to_csv(f)
                            f.write("\n")
                except KeyError:
                    pass


