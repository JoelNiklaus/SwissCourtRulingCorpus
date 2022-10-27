"""
- [x] ROUGE-L, ROUGE-1, ROUGE-2 (Lin, 2004) https://pypi.org/project/rouge-score/
- [x] BLEU (Papineni et al., 2001) (unigram and bigram averaging) https://www.journaldev.com/46659/bleu-score-in-python
- [x] METEOR (Lavie and Agarwal, 2007) https://stackoverflow.com/questions/63778133/how-can-i-implement-meteor-score-when-evaluating-a-model-when-using-the-meteor-s
- [x] Jaccard Similarity, Jaccard distance https://pyshark.com/jaccard-similarity-and-jaccard-distance-in-python/
- [x] Overlap Maximum and Overlap Minimum https://www.geeksforgeeks.org/maximum-number-of-overlapping-intervals/
- BARTScore (Yuan et al., 2021) https://github.com/neulab/BARTScore
- [x] BERTScore Zhang et al. (2020) https://pypi.org/project/bert-score/
- By looking at the annotated sentences themselves and at the reasoning in the free-text annotation for some of the more complex cases4 a qualitative analysis of
the annotation is also possible.
"""
import ast
import os
# Load environment variable from .env
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
import pandas as pd
import copy
from sklearn.metrics import cohen_kappa_score
import nltk
from nltk.translate.bleu_score import sentence_bleu
nltk.download('punkt')
nltk.download('wordnet')
import quantitative_analysis
import qualitative_analysis
import preprocessing


import rouge_score.rouge_scorer as rs
from bert_score import score




LANGUAGES = ast.literal_eval(os.getenv("LANGUAGES"))
PERSONS = ast.literal_eval(os.getenv("PERSONS"))
LABELS = ast.literal_eval(os.getenv("LABELS"))
NAN_KEY = preprocessing.NAN_KEY
AGGREGATIONS = preprocessing.AGGREGATIONS
def process_dataset(datasets: dict, lang: str, version: str):
    """
    Gets language spans and token Dataframes.
    @todo finish documentation

    """
    annotations = datasets["annotations_{}".format(lang)][
        datasets["annotations_{}".format(lang)]["answer"] == "accept"]
    annotations_spans = preprocessing.extract_values_from_column(annotations, "spans", "tokens")
    annotations_tokens = preprocessing.extract_values_from_column(annotations, "tokens", "spans")
    for label in LABELS:
        label_list = []
        label_df = preprocessing.get_span_df(annotations_spans, annotations_tokens, label, lang)[0]
        ws_df = preprocessing.get_white_space_dicts(label_df, "annotations_{}".format(lang))
        label_df = preprocessing.get_tokens_dict(label_df, "tokens_id", "tokens_text", "tokens_dict")
        label_df = label_df.join(ws_df[[f"annotations_{lang}", 'tokens_ws_dict']].set_index(f"annotations_{lang}"),
                                 on="annotations_{}".format(lang))
        for pers in PERSONS:
            label_pers_df = get_annotator_df(pers, label_df, lang, version)
            label_list.append(label_pers_df)

        label_df = merge_label_df(label_list, PERSONS, lang)
        label_df = get_normalize_tokens_dict(label_df)
        if lang == "de":
            for pers in PERSONS:
                label_df = normalize_person_tokens(label_df, pers, lang)
                label_df = preprocessing.string_to_dict(label_df,f'tokens_ws_dict_{pers}')
                label_df = preprocessing.string_to_dict(label_df, f'tokens_dict_{pers}')

            print("Calculating scores...")
            r, be, m, b = calculate_text_scores(label_df, lang)
            score_df_list =[calculate_overlap_min_max(label_df, lang),
                            calculate_jaccard_similarity_distance(label_df, lang), r ,be, m, b]
            for score_df in score_df_list:
                label_df = label_df.merge(score_df, on=f'annotations_{"de"}',
                                      how='outer')
            globals()[f"{label.lower().replace(' ', '_')}_{lang}"] = label_df


            for agg in AGGREGATIONS:
                apply_aggregation(globals()[f"{label.lower().replace(' ', '_')}_{lang}"], "overlap_maximum",
                                  agg)
                apply_aggregation(globals()[f"{label.lower().replace(' ', '_')}_{lang}"], "overlap_minimum",
                                  agg)
                apply_aggregation(globals()[f"{label.lower().replace(' ', '_')}_{lang}"], "jaccard_similarity",
                                  agg)
                apply_aggregation(globals()[f"{label.lower().replace(' ', '_')}_{lang}"], "jaccard_distance",
                                  agg)

                apply_aggregation(globals()[f"{label.lower().replace(' ', '_')}_{lang}"],"meteor_score",
                                  agg)
                apply_aggregation(globals()[f"{label.lower().replace(' ', '_')}_{lang}"],"bleu_score",
                                  agg)

        else:
            globals()[f"{label.lower().replace(' ', '_')}_{lang}"] = label_df




        preprocessing.write_csv(Path("{}/{}_{}.csv".format(lang, f"{label.lower().replace(' ', '_')}_{lang}", version)),
                  globals()[f"{label.lower().replace(' ', '_')}_{lang}"])
        print("Saved {}_{}.csv successfully!".format(f"{label.lower().replace(' ', '_')}_{lang}", version))


def get_annotator_df(annotator_name: str, tokens: pd.DataFrame, lang: str, version: str) -> pd.DataFrame:
    """
    Copies entries from Dataframe from specific annotator.
    Groups tokens_text, tokens_id, tokens_dict from one case together.
    Creates Dataframe containing ids, 'tokens_text','tokens_id', 'tokens_dict'.
    Drops duplicates
    Transforms tokens_id string to list.
    Returns Dataframe
    """
    if version == "1":
        annotator = tokens[
        tokens['_annotator_id'] == f"annotations_{lang}-{annotator_name}"].drop_duplicates().copy()
    if version == "2":
        annotator = tokens[
            tokens['_annotator_id'] == f"annotations_{lang}_inspect-{annotator_name}"].drop_duplicates().copy()
    if version == "3":
        annotator = tokens[
            tokens['_annotator_id'] == f"annotations_{lang}_inspect-{annotator_name}"].drop_duplicates().copy()
        annotator = annotator.append(tokens[
        tokens['_annotator_id'] == f"annotations_{lang}-{annotator_name}"].drop_duplicates().copy())

    annotator = preprocessing.group_columns(annotator, lang)
    annotator = annotator[[f'annotations_{lang}', 'tokens_text', 'tokens_id', 'tokens_dict', 'tokens_ws_dict']]
    annotator = annotator.drop_duplicates()
    annotator["tokens_id"] = annotator["tokens_id"].astype(str).str.split(",")
    no_duplicates = []
    for lst in annotator["tokens_id"].values:
        lst = list(dict.fromkeys(lst))
        no_duplicates.append(lst)
    annotator["tokens_id"] = no_duplicates
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
    merged_df = pd.merge(df_list[i], df_list[i + 1], on=f"annotations_{lang}",
                         suffixes=(f'_{person_suffixes[i]}', f'_{person_suffixes[i+1]}'),
                         how="outer").fillna("Nan")

    return pd.merge(merged_df, df_list[i + 2], on=f"annotations_{lang}", how="outer").fillna("Nan").rename(
        columns={"tokens_text": f"tokens_text_{person_suffixes[i + 2]}",
                 "tokens_id": f"tokens_id_{person_suffixes[i + 2]}",
                 "tokens_dict": f"tokens_dict_{person_suffixes[i + 2]}",
                 'tokens_ws_dict': f'tokens_ws_dict_{person_suffixes[i + 2]}'})


def get_normalize_tokens_dict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Copies Dataframe and joins annotators tokens_dict.
    Creates a token dictionary from the joined individual token dictionaries where each word token has a different value.
    Adds normalized tokens dictionaries to dataframe.
    Returns Dataframe.
    """
    normalized_tokens = []
    df_copy = df.copy()
    for token_dicts in df_copy[
        [f"tokens_dict_{PERSONS[0]}", f"tokens_dict_{PERSONS[1]}", f"tokens_dict_{PERSONS[2]}"]].values:
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


def normalize_person_tokens(df: pd.DataFrame, pers: str, lang: str) -> pd.DataFrame:
    """
    Extracts tokens_text for given person and tokenizes list.
    Gets token dictionary of all persons from Dataframe and adds "Nan" value.
    Asserts that there are no duplicated values in dict after adding "Nan".
    Gets id for each token in persons token list and appends it to normalized_tokens_row.
    Appends normalized_tokens_row to normalized_tokens which creates new column in Dataframe.
    Drops duplicates and returns Dataframe.
    """
    normalized_tokens = []
    for sentences in df[[f"annotations_{lang}", f"tokens_dict_{pers}"]].values:
        normalized_tokens_row = []
        tokens = [sentences[1]]
        if sentences[1] != "Nan":
            tokens = eval(sentences[1]).values()
        token_dict = df[df[f"annotations_{lang}"] == sentences[0]]["normalized_tokens_dict"].values[0]
        token_dict["Nan"] = NAN_KEY
        assert len(token_dict.values()) == len(set(token_dict.values()))
        for word in tokens:
            normalized_tokens_row.append(token_dict[word])
        normalized_tokens.append(normalized_tokens_row)

    df[f"normalized_tokens_{pers}"] = normalized_tokens
    df = df.loc[df.astype(str).drop_duplicates().index]
    return df


def calculate_overlap_min_max(df: pd.DataFrame, lang: str) -> pd.DataFrame:
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
    for value_list in df.copy()[[f"annotations_{lang}", f"normalized_tokens_{PERSONS[0]}", f"normalized_tokens_{PERSONS[1]}",
                          f"normalized_tokens_{PERSONS[2]}", 'normalized_tokens_dict']].values:
        overlap_min_max = {f"annotations_{lang}": value_list[0], "overlap_maximum": [],
                           "overlap_minimum": []}
        combinations = preprocessing.get_combinations(value_list[1:-1],2)
        for comb in combinations:
            overlap_list = []
            comb = sorted(comb, key=len)
            len_min_comb, len_max_comb = len(comb[0]), len(comb[1])
            j = 1
            while j <= len(comb[0]):
                if ''.join(str(i) for i in comb[0][:j]) in ''.join(str(i) for i in comb[1]):
                    j += 1
                    overlap_list.append(comb[0][:j])
                # Section is finished, slice list and check again
                else:
                    comb[0] = comb[0][j:]
                    j = 1
            if len(overlap_list) == 0 or comb == [[NAN_KEY], [NAN_KEY]]:
                overlap_min_max["overlap_maximum"] += [0]
                overlap_min_max["overlap_minimum"] += [0]
            else:
                overlap_max = max(len(elem) for elem in overlap_list)
                assert overlap_max <= len_min_comb
                overlap_min_max["overlap_maximum"] += [overlap_max / len_max_comb]
                overlap_min_max["overlap_minimum"] += [overlap_max / len_min_comb]
        overlap_min_max_list.append(overlap_min_max)

    return pd.DataFrame.from_records(overlap_min_max_list)


def calculate_jaccard_similarity_distance(df: pd.DataFrame, lang) -> pd.DataFrame:
    jaccard_list = []
    for value_list in df.copy()[[f'annotations_{lang}', f"normalized_tokens_{PERSONS[0]}", f"normalized_tokens_{PERSONS[1]}",
                          f"normalized_tokens_{PERSONS[2]}", 'normalized_tokens_dict']].values:
        jaccard = {f'annotations_{lang}': value_list[0], "jaccard_similarity": [], "jaccard_distance": []}
        value_list[1:-1] = normalize_list_length(value_list[1:-1], value_list[-1])
        combinations = preprocessing.get_combinations(value_list[1:-1],2)
        for comb in combinations:
            set_1, set_2 = set(list(comb[0])), set(list(comb[1]))
            # Find intersection of two sets
            nominator_1 = set_1.intersection(set_2)
            # Find symmetric difference of two sets
            nominator_2 = set_1.symmetric_difference(set_2)
            # Find union of two sets
            denominator = set_1.union(set_2)
            # Take the ratio of sizes
            jaccard["jaccard_similarity"].append(len(nominator_1) / len(denominator))
            jaccard["jaccard_distance"].append(len(nominator_2) / len(denominator))
        jaccard_list.append(jaccard)
    return pd.DataFrame.from_records(jaccard_list)

def calculate_rouge_score(i: int,text_combinations: list, rouge_list:list, lang: str) -> list:
    rouge_scores = ['rouge1', 'rouge2', 'rougeL']
    scorer = rs.RougeScorer(rouge_scores, use_stemmer=True)
    rouge = {f'annotations_{lang}': i, rouge_scores[0]: [], rouge_scores[1]: [],
             rouge_scores[2]: []}
    for comb in text_combinations:
        scores = scorer.score(comb[0], comb[1])
        for i in range(len(rouge_scores)):
            rouge[rouge_scores[i]].append(scores[rouge_scores[i]])
    if len(text_combinations) != 0:
        rouge_list.append(rouge)
    return rouge_list

def calculate_bleu_score(i: int,text_combinations:list, bleu_list:list, lang: str):
    bleu = {f'annotations_{lang}': i, "bleu_score": []}
    for comb in text_combinations:
        b_s = sentence_bleu([comb[0]], comb[1])
        bleu["bleu_score"].append(b_s)
    bleu_list.append(bleu)
    return bleu_list


def calculate_meteor_score(i: int,text_combinations:list, meteor_list:list, lang: str):
    meteor = {f'annotations_{lang}': i, "meteor_score":[] }
    for comb in text_combinations:
        m_s = nltk.translate.meteor_score.meteor_score([comb[0]], comb[1])
        meteor["meteor_score"].append(m_s)
    meteor_list.append(meteor)
    return meteor_list


def calculate_bert_score(i: int,text_combinations:list, bert_list:list, lang: str) -> list:
    bert = {f'annotations_{lang}': i, "P": [], "R": [], "F1": []}
    for comb in text_combinations:
        P, R, F1 = score([comb[0]], [comb[1]], lang="other", verbose=True)
        bert["P"].append(P)
        bert["R"].append(R)
        bert["F1"].append(F1)
    #plot_example(text_combinations[0][0], text_combinations[0][1], lang="other")
    bert_list.append(bert)
    return bert_list

def calculate_text_scores(df: pd.DataFrame, lang:str)-> (pd.DataFrame,pd.DataFrame):
    bert_list = []
    meteor_list =[]
    rouge_list = []
    bleu_list = []
    for value_list in df[
        [f'annotations_{lang}', f"tokens_id_{PERSONS[0]}", f"tokens_id_{PERSONS[1]}", f"tokens_id_{PERSONS[2]}",
         f'tokens_dict_{PERSONS[0]}', f'tokens_ws_dict_{PERSONS[0]}',
         f'tokens_dict_{PERSONS[1]}', f'tokens_ws_dict_{PERSONS[1]}',
         f'tokens_dict_{PERSONS[2]}', f'tokens_ws_dict_{PERSONS[2]}']].values:
        text_combinations = get_text_combinations(copy.deepcopy(value_list[1:4]), copy.deepcopy(value_list))
        rouge_list = calculate_rouge_score(value_list[0],text_combinations, rouge_list, lang)
        bert_list = calculate_bert_score(value_list[0], text_combinations, bert_list, lang)
        meteor_list = calculate_meteor_score(value_list[0],text_combinations,meteor_list, lang)
        bleu_list = calculate_bleu_score(value_list[0],text_combinations,bleu_list, lang)
    return pd.DataFrame.from_records(rouge_list),pd.DataFrame.from_records(bert_list), pd.DataFrame.from_records(meteor_list), pd.DataFrame.from_records(bleu_list)

def calculate_cohen_kappa(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """
    Gets value_list containing all normalized token lists per id for a language.
    Creates dictionary and gets combinations of the token value_lists.
    For each combination of two lists normalizes list length
    and calculates the cohen kappa score using sklearn.metrics cohen_kappa_score.
    Adds the cappa scores to list and to row of Dataframe.
    Returns Dataframe.
    """

    cohen_kappa_scores_list = []
    cohen_kappa_scores = {}
    for value_list in df[[f'annotations_{lang}', f"normalized_tokens_{PERSONS[0]}", f"normalized_tokens_{PERSONS[1]}",
                                f"normalized_tokens_{PERSONS[2]}", 'normalized_tokens_dict']].values:
        cohen_kappa_scores = {f'annotations_{lang}': value_list[0], "cohen_kappa_scores": []}
        combinations = preprocessing.get_combinations(value_list[1:-1],2)
        for comb in combinations:
            if len(comb[0]) != 1 and len(comb[1]) != 1:
                list_a, list_b = normalize_list_length(combinations, value_list[-1])
                cohen_kappa_scores["cohen_kappa_scores"] += [cohen_kappa_score(list_a, list_b)]
    cohen_kappa_scores_list.append(cohen_kappa_scores)

    return pd.DataFrame.from_records(cohen_kappa_scores_list)

def get_text(token_list: list, tokens_dict: dict,ws_dict: dict)->str:
    text = ""
    for nr in token_list:
        if nr != NAN_KEY:
            if ws_dict[int(nr)]:
                text = text + tokens_dict[int(nr)] + " "
            else:
                text = text + tokens_dict[int(nr)]
    return text

def get_text_combinations(token_list: list, token_dict_list: list)-> list:
    text_list = []
    dict_indexes = {PERSONS[0]: [4, 5], PERSONS[1]: [6, 7], PERSONS[2]: [8, 9]}
    for i in range(0, len(token_list)):
        if type(token_list[i]) != list:
            token_list[i] = [token_list[i]]
        else:
            token_list[i].append(PERSONS[i])
    combinations = preprocessing.get_combinations(token_list, 2)
    for comb in combinations:
        token_ws_dict_1, token_ws_dict_2 = dict_indexes[comb[0][-1]], dict_indexes[comb[1][-1]]
        txt_1 = get_text(comb[0][:-1], token_dict_list[token_ws_dict_1[0]], token_dict_list[token_ws_dict_1[1]])
        txt_2 = get_text(comb[1][:-1], token_dict_list[token_ws_dict_2[0]], token_dict_list[token_ws_dict_2[1]])
        text_list.append((txt_1, txt_2))
    return text_list

def normalize_list_length(list_of_list: list, token_dict: dict) -> (list, list):
    """
    Appends "Nan" to normalize list length (make them same length).
    Returns lists.
    """
    max_length = find_max_length(list_of_list)[1]
    for lst in list_of_list:
        index = list(list_of_list).index(lst)
        if NAN_KEY not in lst:
            while len(lst) < max_length:
                lst.append(token_dict["Nan"])
            list_of_list[index] = lst

    return list_of_list


def find_max_length(lst: list) -> (list, int):
    max_list = max(x for x in lst)
    max_length = max(len(x) for x in lst)

    return max_list, max_length


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
    extracted_datasets_1 = preprocessing.extract_dataset("../legal_expert_annotations/{}/annotations_{}.jsonl", "../legal_expert_annotations/{}/annotations_{}-{}.jsonl")
    extracted_datasets_2 = preprocessing.extract_dataset("../legal_expert_annotations/{}/annotations_{}_inspect.jsonl", "../legal_expert_annotations/{}/annotations_{}_inspect-{}.jsonl")
    extracted_datasets_3 = preprocessing.extract_dataset("../legal_expert_annotations/{}/annotations_{}_merged.jsonl",
                                                         "../legal_expert_annotations/{}/annotations_{}_merged-{}.jsonl")

    for l in LANGUAGES:
        try:
            for triple in [(extracted_datasets_1, l, "1"),(extracted_datasets_2, l,"2"),(extracted_datasets_3, l,"3")]:
                 process_dataset(triple[0],triple[1],triple[2])
        except KeyError as err:
             print(err)
             pass

    quantitative_analysis.analysis()
    qualitative_analysis.analysis()
