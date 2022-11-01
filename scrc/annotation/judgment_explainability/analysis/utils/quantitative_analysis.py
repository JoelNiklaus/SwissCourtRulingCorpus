import ast
from nltk.tokenize import word_tokenize
from pathlib import Path
import pandas as pd
import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing

LANGUAGES = ["de", "fr", "it"]
PERSONS = ["angela", "lynn", "thomas"]
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]

AGGREGATIONS = ["mean", "max", "min"]
GOLD_SESSION = {"de": "gold_final", "fr": "gold_nina", "it": "gold_nina"}
LEGAL_AREAS = ["penal_law", "social_law", "civil_law"]
YEARS = ["2015", "2016", "2017", "2018", "2019", "2020"]
DF_LIST = []
EXTRACTED_DATASETS_GOLD = preprocessing.extract_dataset(
        "../legal_expert_annotations/{}/gold/gold_annotations_{}.jsonl",
        "../legal_expert_annotations/{}/gold/gold_annotations_{}-{}.jsonl")
EXTRACTED_DATASETS_MERGED = preprocessing.extract_dataset("../legal_expert_annotations/{}/annotations_{}_merged.jsonl",
                                                         "../legal_expert_annotations/{}/annotations_{}_merged-{}.jsonl")

OCCLUSION_PATHS = {"test_sets": ("../occlusion/lower_court_test_sets/{}/lower_court_test_set_{}.csv",
                                 "../occlusion/occlusion_test_sets/{}/occlusion_test_set_{}_exp_{}.csv"),
                   "prediction": ("../occlusion/lower_court_predictions/{}/predictions_test.csv",
                                  "../occlusion/occlusion_predictions/{}_{}/predictions_test.csv")}
NUMBER_OF_EXP = [1, 2, 3, 4]
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


"""
Hier betonen das es um akzeptierte fälle von gold standard geht
"""

def count_is_correct(dataset: pd.DataFrame):
    annotations = dataset[dataset["answer"] == "accept"]
    return len(annotations[annotations["is_correct"] == True].groupby("id_csv"))


def apply_aggregation(annotations: pd.DataFrame, col_name: str, aggregation: str):
    if aggregation == "mean":
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
    annotations = pd.DataFrame()
    for la in LEGAL_AREAS:
        annotations[la] = df[la].dropna()

    return annotations


def get_total_agg(annotations: pd.DataFrame, aggregation: str, col_name: str):
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
    return la, year


def count_fact_length(dataset: pd.DataFrame):
    annotations = dataset[dataset["answer"] == "accept"]
    annotations["text"] = annotations["text"].str.len()
    annotations["tokens"] = annotations["tokens"].map(len)
    for agg in AGGREGATIONS:
        char = get_agg_table(apply_aggregation(annotations, "text", agg), "text")
        char = char.append(get_total_agg(annotations, agg, "text")[0])
        char[f"{agg}_year"] = get_total_agg(annotations, agg, "text")[1]
        char.index.name = f"char_{agg}"
        tokens = get_agg_table(apply_aggregation(annotations, "tokens", agg), "tokens")
        tokens = tokens.append(get_total_agg(annotations, agg, "tokens")[0])
        tokens[f"{agg}_year"] = get_total_agg(annotations, agg, "tokens")[1]
        tokens.index.name = f"tokens_{agg}"
        DF_LIST.append(char)
        DF_LIST.append(tokens)



def count_tokens_per_label(lang: str):
    pers_mean = []
    label_df_list = []
    label_mean = []
    for label in LABELS:
        label_df = preprocessing.read_csv(f"{lang}/{label.lower().replace(' ', '_')}_{lang}_3.csv", "index")
        label_df["length"] = 0
        for person in PERSONS:
            try:
                label_df[f"length_{person}"] = label_df[f"tokens_id_{person}"].apply(lambda x: get_length(x))
                label_df["length"].append(label_df[f"length_{person}"])
                pers_mean.append("{},{},{}\n".format(label, person, label_df[f"length_{person}"].mean()))
                label_df_list = label_df_list + list(label_df[f"length_{person}"].values)
            except KeyError:
                pass
        label_mean.append("{},{}\n".format(label, sum(label_df_list) / len(label_df_list)))
    return label_mean, pers_mean

def count_occlusion_tokens(lang: str):
    lower_court_test_set = preprocessing.read_csv(OCCLUSION_PATHS["test_sets"][0].format(lang, lang), "index")
    for nr in NUMBER_OF_EXP:
        occlusion_test_set = preprocessing.read_csv(OCCLUSION_PATHS["test_sets"][1].format(lang, lang, nr), "index")






def get_length(string: str):
    if string != "Nan":
        return len(ast.literal_eval(string))
    else:
        return 0


def write_from_list(f, lst: list):
    for string in lst:
        f.write(string)
    f.write("\n")


def write_csv(path: Path, counts: pd.DataFrame, label_mean: list, pers_mean: list):
    with open(path, "w") as f:
        f.truncate()
        counts.to_csv(f)
        f.write("\n")
        write_from_list(f, label_mean)
        write_from_list(f, pers_mean)
        for df in DF_LIST:
            df.to_csv(f)
            f.write("\n")




"""
Hier darauf eingehen, dass das deutsche Datenset am unausgeglichesten ist weil am meisten daran gearbeitet wurde.
Man hat mit balanciertem set gestarten nach annotationen etwas weniger balanciert, wollte nicht manuell fälle herauswerfen.
gold standrad noch abeglenht -> time constraint, proof of concept für zukünftige arbeiten ein ziel die ausgeglichenheit zu verfeinern

Bei Annotations übersicht nr verwenden und sagen latest annotation before GS.

"""


def annotation_analysis():
    for l in LANGUAGES:
        count_fact_length(EXTRACTED_DATASETS_GOLD[f"annotations_{l}-{GOLD_SESSION[l]}"])
        counts = pd.DataFrame({"total_number_of_cases": [
            count_total_cases(EXTRACTED_DATASETS_GOLD[f"annotations_{l}-{GOLD_SESSION[l]}"])],
            "number_of_comments": [
                count_user_input(EXTRACTED_DATASETS_MERGED[f"annotations_{l}"])],
            "number_of_not_accepted": [count_not_accept(
                EXTRACTED_DATASETS_GOLD[f"annotations_{l}-{GOLD_SESSION[l]}"])],
            "number_of_is_correct": [count_is_correct(
                EXTRACTED_DATASETS_GOLD[f"annotations_{l}-{GOLD_SESSION[l]}"])]})
        label_mean, pers_mean = count_tokens_per_label(l)
        write_csv(Path(f"{l}/quantitative/quantitative_analysis_{l}.csv"), counts, label_mean, pers_mean)

def occlusion_analysis():
    for l in LANGUAGES:
        count_fact_length(EXTRACTED_DATASETS_GOLD[f"annotations_{l}-{GOLD_SESSION[l]}"])



