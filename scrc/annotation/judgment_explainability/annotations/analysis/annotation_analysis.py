"""
- Krippendorff’s Alpha
- ROUGE-L, ROUGE-1, ROUGE-2 (Lin, 2004)
- BLEU (Papineni et al., 2001) (unigram and bigram averaging)
- METEOR (Lavie and Agarwal, 2007)
- Jaccard Similarity
- Overlap Maximum and Overlap Minimum
- BARTScore (Yuan et al., 2021) and BERTScore Zhang et al. (2020).
- By looking at the annotated sentences themselves and at the reasoning in the free-text annotation for some of the more complex cases4 a qualitative analysis of
the annotation is also possible.

Sources for Metrics:
- https://www.statsmodels.org/stable/_modules/statsmodels/stats/inter_rater.html
- https://pypi.org/project/rouge-score/
- https://www.journaldev.com/46659/bleu-score-in-python
- https://stackoverflow.com/questions/63778133/how-can-i-implement-meteor-score-when-evaluating-a-model-when-using-the-meteor-s
- https://pypi.org/project/bert-score/
- https://github.com/neulab/BARTScore
- https://pyshark.com/jaccard-similarity-and-jaccard-distance-in-python/
- https://www.geeksforgeeks.org/maximum-number-of-overlapping-intervals/





"""
import itertools
import json
import re
from pathlib import Path
from random import randint

import nltk
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from scrc.annotation.judgment_explainability.annotations.analysis.preprocessing_functions import LANGUAGES, LANGUAGES_NLTK, PERSONS,\
LABELS, AGGREGATIONS,  extract_dataset,dump_user_input,dump_case_not_accepted,to_csv ,get_tokens_dict, extract_values_from_column, get_span_df, \
    group_columns
nltk.download('punkt')


# Sets pandas print options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)






def process_dataset(datasets: dict, lang: str):
    """
    Gets language spans and token Dataframes.
    @todo finish documentation

    """
    annotations = datasets["annotations_{}".format(lang)][
        datasets["annotations_{}".format(lang)]["answer"] == "accept"]
    annotations_spans = extract_values_from_column(annotations, "spans", "tokens")
    annotations_tokens = extract_values_from_column(annotations, "tokens", "spans")
    for label in LABELS:
        label_list = []
        label_df = get_span_df(annotations_spans, annotations_tokens, label, lang)[0]
        label_df = get_tokens_dict(label_df,"tokens_id","tokens_text", "tokens_dict")
        for pers in PERSONS:
            globals()[f"{label.lower().replace(' ', '_')}_{lang}_{pers}"] = get_annotator_df(pers, label_df,
                                                                                             lang)
            label_list.append(globals()[f"{label.lower().replace(' ', '_')}_{lang}_{pers}"])
        globals()[f"{label.lower().replace(' ', '_')}_{lang}"] = merge_label_df(label_list, PERSONS, lang)

        globals()[f"{label.lower().replace(' ', '_')}_{lang}"] = get_normalize_tokens_dict(
            globals()[f"{label.lower().replace(' ', '_')}_{lang}"])
        if lang == "de":
            for pers in PERSONS:
                globals()[f"{label.lower().replace(' ', '_')}_{lang}"] = normalize_person_tokens(
                    globals()[f"{label.lower().replace(' ', '_')}_{lang}"], pers, lang, LANGUAGES_NLTK[lang])

            o = calculate_overlap_min_max(globals()[f"{label.lower().replace(' ', '_')}_{lang}"], lang)
            globals()[f"{label.lower().replace(' ', '_')}_{lang}"] = globals()[
                f"{label.lower().replace(' ', '_')}_{lang}"].merge(o, on='annotations_{}'.format("de"),
                                                                   how='outer')

            for agg in AGGREGATIONS:
                apply_aggregation(globals()[f"{label.lower().replace(' ', '_')}_{lang}"], "overlap_maximum",
                                  agg)
                apply_aggregation(globals()[f"{label.lower().replace(' ', '_')}_{lang}"], "overlap_minimum",
                                  agg)
            to_csv(Path("{}/{}.csv".format(lang,f"{label.lower().replace(' ', '_')}_{lang}")),
                   globals()[f"{label.lower().replace(' ', '_')}_{lang}"])
            print("Saved {}.csv successfully!".format(f"{label.lower().replace(' ', '_')}_{lang}"))

def get_annotator_df(annotator_name: str, tokens: pd.DataFrame, lang: str) -> pd.DataFrame:
    """
    Copies entries from Dataframe from specific annotator.
    Groups tokens_text, tokens_id, tokens_dict from one case together.
    Creates Dataframe containing ids, 'tokens_text','tokens_id', 'tokens_dict'.
    Drops duplicates
    Transforms tokens_id string to list.
    Returns Dataframe
    """

    annotator = tokens[
        tokens['_annotator_id'] == "annotations_{}-{}".format(lang, annotator_name)].drop_duplicates().copy()
    annotator = group_columns(annotator, lang)
    annotator = annotator[['annotations_{}'.format(lang), 'tokens_text', 'tokens_id','tokens_dict']]
    annotator = annotator.drop_duplicates()
    annotator["tokens_id"] = annotator["tokens_id"].astype(str).str.split(",")
    return annotator



def remove_duplicate(list_duplicate: list) -> list:
    """
    Removes duplicates from list, returns list
    """
    return list(dict.fromkeys(list_duplicate))


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
                 "tokens_id": "tokens_id_{}".format(person_suffixes[i + 2]),
                 "tokens_dict":"tokens_dict_{}".format(person_suffixes[i + 2])})
    return df


def get_normalize_tokens_dict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Copies Dataframe and joins annotators tokens_dict.
    Creates a token dictionary from the joined individual token dictionaries where each word token has a different value.
    Adds normalized tokens dictionaries to dataframe.
    Returns Dataframe.
    """
    normalized_tokens = []
    df_copy = df.copy()
    for token_dicts in df_copy[[f"tokens_dict_{PERSONS[0]}", f"tokens_dict_{PERSONS[1]}", f"tokens_dict_{PERSONS[2]}"]].values:
        tokens = []
        for token_dict in token_dicts:
            if token_dict != "Nan":
                for token in eval(token_dict).values():
                    tokens.append(token)
            if token_dict == "Nan":
                tokens.append(token_dict)
        normalized_tokens.append(dict(zip(tokens, range(0, len(tokens)))))
    df["normalized_tokens_dict"] = normalized_tokens
    return df


def normalize_person_tokens(df: pd.DataFrame, pers: str, lang: str, lang_nltk: str) -> pd.DataFrame:
    """
    Extracts tokens_text for given person and tokenizes list.
    Gets token dictionary of all persons from Dataframe and adds "Nan" value.
    Asserts that there are no duplicated values in dict after adding "Nan".
    Gets id for each token in persons token list and appends it to normalized_tokens_row.
    Appends normalized_tokens_row to normalized_tokens which creates new column in Dataframe.
    Drops duplicates and returns Dataframe.
    """
    normalized_tokens = []
    for sentences in df[["annotations_{}".format(lang), "tokens_dict_{}".format(pers)]].values:
        normalized_tokens_row = []
        tokens = [sentences[1]]
        if sentences[1] != "Nan":
            tokens = eval(sentences[1]).values()
        token_dict = df[df["annotations_{}".format(lang)] == sentences[0]]["normalized_tokens_dict"].values[0]
        token_dict["Nan"] = 10000
        assert len(token_dict.values()) == len(set(token_dict.values()))
        for word in tokens:
            normalized_tokens_row.append(token_dict[word])
        normalized_tokens.append(normalized_tokens_row)

    df["normalized_tokens_{}".format(pers)] = normalized_tokens
    df = df.loc[df.astype(str).drop_duplicates().index]
    print(df)
    return df


def calculate_overlap_min_max(label_df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """
    Gets value_list containing all normalized token lists per id for a language.
    Creates dictionary and gets combinations of the token value_lists.
    For each combination of two lists finds the maximal overlapping sequence (e.g. [1,2,3] and [2,3,4] -> [2,3]).
    Asserts max length is less than or equal to smallest sample (maximum of overlapping section is section itself).
    Calculates the overlapping maximum and minimum score using the length of this sequence divided by the maximum or minimum of the sample sets.
    If there is no overlap or the sample content is Nan ([10000]) the overlap_maximum and overlap_minimum equals 0.
    Adds the overlap_maximum and overlap_minimum scores to the dict and adds this dict to a list.
    Creates a DataFrame from the list and returns it.
    """
    overlap_min_max_list = []
    for value_list in label_df[['annotations_{}'.format(lang), "normalized_tokens_angela", "normalized_tokens_lynn",
                                "normalized_tokens_thomas", 'normalized_tokens_dict']].values:
        overlap_min_max = {'annotations_{}'.format(lang): value_list[0], "overlap_maximum": [],
                           "overlap_minimum": []}
        combinations = get_combinations(value_list[1:-1])
        for comb in combinations:
            overlap_list = []
            comb = sorted(comb, key=len)
            len_min_comb, len_max_comb = len(comb[0]), len(comb[1])
            i = 1
            while i <= len(comb[0]):
                if ''.join(str(i) for i in comb[0][:i]) in ''.join(str(i) for i in comb[1]):
                    i = i + 1
                    overlap_list.append(comb[0][:i])
                # Section is finished, slice list and check again
                else:
                    comb[0] = comb[0][i:]
                    i = 1
            if len(overlap_list) == 0 or comb == [[10000], [10000]]:
                overlap_min_max["overlap_maximum"] += [0]
                overlap_min_max["overlap_minimum"] += [0]
            else:
                overlap_max = max(len(elem) for elem in overlap_list)
                assert overlap_max <= len_min_comb
                overlap_min_max["overlap_maximum"] += [overlap_max / len_max_comb]
                overlap_min_max["overlap_minimum"] += [overlap_max / len_min_comb]
        overlap_min_max_list.append(overlap_min_max)
    overlap_min_max_df = pd.DataFrame.from_records(overlap_min_max_list)

    return overlap_min_max_df


def calculate_cohen_kappa(label_df: pd.DataFrame, lang: str) -> pd.DataFrame:
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


def apply_aggregation(df: pd.DataFrame, column_name, aggregation: str):
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
    extracted_datasets = extract_dataset("../{}/annotations_{}.jsonl","../{}/annotations_{}-{}.jsonl")
    #dump_user_input(extracted_datasets)
    #dump_case_not_accepted(extracted_datasets)

    for l in LANGUAGES:
        try:
            process_dataset(extracted_datasets, l)
        except KeyError as err:
            print(err)
            pass
