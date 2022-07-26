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
import itertools
import json
from pathlib import Path
from random import randint

import nltk
import pandas
import pandas as pd
from sklearn.metrics import cohen_kappa_score

nltk.download('punkt')

LANGUAGES = ["de", "fr", "it"]
LANGUAGES_NLTK = {"de": "german", "fr": "french", "it": "italian"}
PERSONS = ["angela", "lynn", "thomas"]
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
AGGREGATIONS = ["mean", "max", "min"]
# Sets pandas print options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def extract_dataset() -> dict:
    """
    Tries to extracts data from JSONL file to a dictionary list.
    Transforms dictionary list to dataframe.
    Catches file not found.
    Returns a dictionary of dataframes with filename as keys.
    """
    datasets = {}
    for language in LANGUAGES:
        a_list = []
        # with open("../test.jsonl".format(language), "r") as json_file:
        with open("../annotations_{}.jsonl".format(language), "r") as json_file:
            json_list = list(json_file)
            for json_str in json_list:
                result = json.loads(json_str)
                a_list.append(result)
                assert isinstance(result, dict)
                # List of dict to dataframe
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

def to_csv(filepath: Path, df: pd.DataFrame):
    """
    Creates a csv from Dataframe
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)


def dump_user_input(dataset_dict: dict):
    """
    Dumps all the user inputs as csv.
    Catches key errors.
    """
    for data in dataset_dict:
        try:
            user_input = dataset_dict[data][dataset_dict[data]["user_input"] != ""]
            to_csv(Path(dataset_dict[data].index.name + "_user_input.csv"),
                   user_input[['id_scrc', '_annotator_id', "user_input"]])
            print("Saved {}.csv successfully!".format(dataset_dict[data].index.name + "_user_input"))
        except KeyError:
            pass




def dump_case_not_accepted(dataset_dict: dict):
    """
    Dumps all the all not accepted cases as csv.
    Catches key errors.
    """
    for data in dataset_dict:
        try:
            case_not_accepted = dataset_dict[data][dataset_dict[data]["answer"] != "accept"]
            to_csv(Path(dataset_dict[data].index.name + "_ig_re.csv"),
                   case_not_accepted[['id_scrc', '_annotator_id', "user_input"]])
            print("Saved {}.csv successfully!".format(dataset_dict[data].index.name + "_ig_re"))
        except KeyError:
            pass




def process_dataset(datasets: dict, lang:str):
    """
    Gets language spans and token Dataframes.


    """
    annotations = datasets["annotations_{}".format(lang)][
        datasets["annotations_{}".format(lang)]["answer"] == "accept"]
    annotations_spans = extract_values_from_column(annotations, "spans", "tokens")
    annotations_tokens = extract_values_from_column(annotations, "tokens", "spans")
    for label in LABELS:
        label_list = []
        label_df = get_span_df(annotations_spans, annotations_tokens, label, lang)
        for pers in PERSONS:
            globals()[f"{label.lower().replace(' ', '_')}_{lang}_{pers}"] = get_annotator_set(pers, label_df,
                                                                                              lang)
            label_list.append(globals()[f"{label.lower().replace(' ', '_')}_{lang}_{pers}"])
        globals()[f"{label.lower().replace(' ', '_')}_{lang}"] = merge_label_df(label_list, PERSONS, lang)
        globals()[f"{label.lower().replace(' ', '_')}_{lang}"] = get_normalize_tokens_dict(
            globals()[f"{label.lower().replace(' ', '_')}_{lang}"])
        if lang == "de":
            for pers in PERSONS:
                globals()[f"{label.lower().replace(' ', '_')}_{lang}"] = normalize_person_tokens(
                    globals()[f"{label.lower().replace(' ', '_')}_{lang}"], pers, lang, LANGUAGES_NLTK[lang])

            # Merges Dataframe with cohen kappa score dataframe
            globals()[f"{label.lower().replace(' ', '_')}_{lang}"] = globals()[
                f"{label.lower().replace(' ', '_')}_{lang}"].merge(
                calculate_cohen_kappa(globals()[f"{label.lower().replace(' ', '_')}_{lang}"]),
                on='annotations_{}'.format("de"),
                how='left')
            for agg in AGGREGATIONS:
                apply_aggregation(globals()[f"{label.lower().replace(' ', '_')}_{lang}"], "cohen_kappa_scores",
                                  agg)

            to_csv(Path("{}.csv".format(f"{label.lower().replace(' ', '_')}_{lang}")),
                   globals()[f"{label.lower().replace(' ', '_')}_{lang}"])
            print("Saved {}.csv successfully!".format(f"{label.lower().replace(' ', '_')}_{lang}"))

def extract_values_from_column(annotations: pandas.DataFrame, col_1: str, col_2: str) -> pandas.DataFrame:
    """
    Extracts values from list of dictionaries in columns (by explode), resets index and drops col_2.
    Extracts dictionaries from row (apply) and adds corresponding prefix.
    Drops original column (col_1).
    Joins full dataframe with new column and returns it.
    """
    annotations_col = annotations.explode(col_1).reset_index().drop([col_2], axis=1)
    df_col = annotations_col[col_1].apply(pd.Series).add_prefix("{}_".format(col_1))
    annotations_col = annotations_col.drop([col_1], axis=1)
    return annotations_col.join(df_col)

def get_span_df(annotations_spans: pandas.DataFrame, annotations_tokens: pandas.DataFrame, span: str, lang: str) -> (
pandas.DataFrame):
    """
    Extract all rows where span_label matches the span given as parameter (e.g. span = "Lower court").
    Queries list of values from chosen rows and creates list of token numbers (token ids) of span start and end number.
    Adds token ids to dict.
    Extracts token ids from dict and gets corresponding word tokens for the ids.
    Appends word tokens and ids as Dataframe to list.
    Returns Dataframe containing ids and words of spans.
    """
    spans = annotations_spans[annotations_spans["spans_label"] == span]
    token_numbers = {}
    for mini_list in list(
            spans[['annotations_{}'.format(lang), '_annotator_id', 'spans_token_start', 'spans_token_end']].values):
        numbers = []
        # Range of numbers between spans_token_start spans_token_end
        for nr in list(range(int(mini_list[2]), int(mini_list[3]) + 1)):
            numbers.append(nr)
        token_numbers["{}.{}.{}".format(mini_list[0], mini_list[1], randint(0, 100000))] = numbers
    spans_list = []
    for key in token_numbers:
        new_annotations_tokens = annotations_tokens[
            annotations_tokens['annotations_{}'.format(lang)] == int(key.split(".")[0])].copy()
        new_annotations_tokens = new_annotations_tokens[new_annotations_tokens["tokens_id"].isin(token_numbers[key])]
        new_annotations_tokens = new_annotations_tokens[new_annotations_tokens['_annotator_id'] == key.split(".")[1]]
        spans_list.append(new_annotations_tokens)
    spans = pd.concat(spans_list)
    return spans

def get_annotator_set(annotator_name: str, tokens: pandas.DataFrame, lang: str) -> pandas.DataFrame:
    """
    Copies entries from Dataframe from specific annotator.
    Groups tokens_text and tokens_id from one case together.
    Creates Dataframe containing ids, 'tokens_text' and 'tokens_id'.
    Drops duplicates
    Transforms tokens_id string to list.
    Returns Dataframe
    """
    annotator = tokens[
        tokens['_annotator_id'] == "annotations_{}-{}".format(lang, annotator_name)].drop_duplicates().copy()
    annotator['tokens_text'] = annotator.groupby(['annotations_{}'.format(lang)])['tokens_text'].transform(
        lambda x: ' '.join(x))
    annotator['tokens_id'] = annotator.groupby(['annotations_{}'.format(lang)])['tokens_id'].transform(
        lambda x: ','.join(x.astype(str)))
    annotator = annotator[['annotations_{}'.format(lang), 'tokens_text', 'tokens_id']]
    annotator = annotator.drop_duplicates()
    annotator["tokens_id"] = annotator["tokens_id"].astype(str).str.split(",")
    return annotator

def remove_duplicate(list_dublicate: list) -> list:
    """
    Removes duplicates from list, returns list
    """
    return list(dict.fromkeys(list_dublicate))


def merge_label_df(df_list: list, person_suffixes: list, lang: str):
    """
    Merges first and second Dataframe using outer join.
    Formats column names using person_suffixes, fills Nan values with "Nan".
    Repeats the same merge with the new merged Dataframe and the third Dataframe.
    @Todo Create merge loop for more dfs
    Returns merged Dataframe.
    """
    i = 0
    merged_df = pd.merge(df_list[i], df_list[i + 1], on="annotations_{}".format(lang),
                         suffixes=('_{}'.format(person_suffixes[i]), '_{}'.format(person_suffixes[i + 1])),
                         how="outer").fillna("Nan")

    df = pd.merge(merged_df, df_list[i + 2], on="annotations_{}".format(lang), how="outer").fillna("Nan").rename(
        columns={"tokens_text": "tokens_text_{}".format(person_suffixes[i + 2]),
                 "tokens_id": "tokens_id_{}".format(person_suffixes[i + 2])})
    return df


def get_normalize_tokens_dict(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Copies Dataframe and joins annotators tokens_text lists.
    Creates a token dictionary from the joined lists where each word token has a different value.
    Adds normalized tokens to dataframe and drops tokens_text column.
    Returns Dataframe.
    """
    normalized_tokens = []
    df_copy = df.copy()
    df["tokens_text"] = df_copy["tokens_text_{}".format(PERSONS[0])].astype(str) + " " + df_copy["tokens_text_{}".format(PERSONS[1])].astype(
        str) + " " + df_copy["tokens_text_{}".format(PERSONS[2])].astype(str)
    for sentence in df["tokens_text"].values:
        tokens = nltk.word_tokenize(sentence, language='german')
        normalized_tokens.append(dict(zip(tokens, range(0, len(tokens)))))
    df["normalized_tokens_dict"] = normalized_tokens
    df = df.drop('tokens_text', axis=1)
    return df


def normalize_person_tokens(df: pandas.DataFrame, pers: str, lang: str, lang_nltk: str) -> pandas.DataFrame:
    """
    Extracts tokens_text for given person and tokenizes list.
    Gets token dictionary of all persons from Dataframe and adds "Nan" value.
    Asserts that there are no duplicated values in dict after adding "Nan".
    Gets id for each token in persons token list and appends it to normalized_tokens_row.
    Appends normalized_tokens_row to normalized_tokens which creates new column in Dataframe.
    Drops duplicates and returns Dataframe.
    """
    normalized_tokens = []
    for sentences in df[["annotations_{}".format(lang), "tokens_text_{}".format(pers)]].values:
        normalized_tokens_row = []
        tokens = nltk.word_tokenize(sentences[1], language=lang_nltk)
        token_dict = df[df["annotations_{}".format(lang)] == sentences[0]]["normalized_tokens_dict"].values[0]
        token_dict["Nan"] = 10000
        assert len(token_dict.values()) == len(set(token_dict.values()))
        for word in tokens:
            normalized_tokens_row.append(token_dict[word])
        normalized_tokens.append(normalized_tokens_row)

    df["normalized_tokens_{}".format(pers)] = normalized_tokens
    df = df.loc[df.astype(str).drop_duplicates().index]
    return df


def calculate_cohen_kappa(label_df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Gets value_list containing all normalized token lists per id for a language.
    Creates dictionary and gets combinations of the token value_lists.
    For each combination of two lists normalizes list length
    and calculates the cohen kappa score using sklearn.metrics cohen_kappa_score.
    Adds the cappa scores to list and to row of Dataframe.
    Returns Dataframe.
    """
    cohen_kappa_scores_list = []
    for value_list in label_df[['annotations_{}'.format(lang), "normalized_tokens_angela", "normalized_tokens_lynn",
                                "normalized_tokens_thomas", 'normalized_tokens_dict']].values:
        for i in range(1, len(value_list) - 2):
            cohen_kappa_scores = {'annotations_{}'.format(lang): value_list[0], "cohen_kappa_scores": []}
            combinations = get_combinations(value_list[1:-1])
            for comb in combinations:
                if len(comb[0]) != 1 and len(comb[1]) != 1:
                    list_a, list_b = normalize_list_length(comb[0], comb[1], value_list[-1])
                    cohen_kappa_scores["cohen_kappa_scores"] += [cohen_kappa_score(list_a, list_b)]
        cohen_kappa_scores_list.append(cohen_kappa_scores)
    cohen_kappa_scores_df = pd.DataFrame.from_records(cohen_kappa_scores_list)
    return cohen_kappa_scores_df

def get_combinations(value_list: list) -> list:
    """
    Gets combinations of a list of values and returns them.
    """
    combinations = []
    for L in range(0, len(value_list) + 1):
        for subset in itertools.combinations(value_list, L):
            if len(subset) == 2:
                combinations.append(subset)

    return combinations
def normalize_list_length(list_1: list, list_2: list, token_dict: dict) -> (list, list):
    """
    Appends "Nan" to normalize list length (make them same length).
    Returns lists.
    """
    while len(list_1) != len(list_2):
        if len(list_1) > len(list_2):
            list_2.append(token_dict["Nan"])
        if len(list_1) < len(list_2):
            list_1.append(token_dict["Nan"])

    return list_1, list_2


def apply_aggregation(df: pandas.DataFrame, column_name, aggregation: str):
    """
    Applies an aggregation function to a column of a dataframe (e.g. to column cohen_kappa_scores).
    Returns Dataframe containing column of aggregation.
    """
    if aggregation == "mean":
        df["{}_{}".format(aggregation, column_name)] = pd.DataFrame(df[column_name].values.tolist()).mean(1)
    if aggregation == "max":
        df["{}_{}".format(aggregation, column_name)] = pd.DataFrame(df[column_name].values.tolist()).max(1)
    if aggregation == "min":
        df["{}_{}".format(aggregation, column_name)] = pd.DataFrame(df[column_name].values.tolist()).min(1)
    return df




if __name__ == '__main__':
    extracted_datasets = extract_dataset()
    dump_user_input(extracted_datasets)
    dump_case_not_accepted(extracted_datasets)


    for l in LANGUAGES:
        try:
            process_dataset(extracted_datasets,l)
        except KeyError as err:
            print(err)
            pass

