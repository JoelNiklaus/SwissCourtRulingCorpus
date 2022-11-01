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

import copy
from pathlib import Path
import nltk
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

nltk.download('punkt')
nltk.download('wordnet')
import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing
import rouge_score.rouge_scorer as rs
from bert_score import score as bert_score

LANGUAGES = ["de", "fr", "it"]
PERSONS = ["angela", "lynn", "thomas"]
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
SCORES = ["overlap_maximum", "overlap_minimum", "jaccard_similarity", "jaccard_distance", "meteor_score", "bleu_score"]
NAN_KEY = preprocessing.NAN_KEY
AGGREGATIONS = preprocessing.AGGREGATIONS


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
    for value_list in df.copy()[
        [f"annotations_{lang}", f"normalized_tokens_{PERSONS[0]}", f"normalized_tokens_{PERSONS[1]}",
         f"normalized_tokens_{PERSONS[2]}", 'normalized_tokens_dict']].values:
        overlap_min_max = {f"annotations_{lang}": value_list[0], "overlap_maximum": [],
                           "overlap_minimum": []}
        combinations = preprocessing.get_combinations(value_list[1:-1], 2)
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
    for value_list in df.copy()[
        [f'annotations_{lang}', f"normalized_tokens_{PERSONS[0]}", f"normalized_tokens_{PERSONS[1]}",
         f"normalized_tokens_{PERSONS[2]}", 'normalized_tokens_dict']].values:
        jaccard = {f'annotations_{lang}': value_list[0], "jaccard_similarity": [], "jaccard_distance": []}
        value_list[1:-1] = preprocessing.normalize_list_length(value_list[1:-1], value_list[-1])
        combinations = preprocessing.get_combinations(value_list[1:-1], 2)
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


def calculate_rouge_score(i: int, text_combinations: list, rouge_list: list, lang: str) -> list:
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


def calculate_bleu_score(i: int, text_combinations: list, bleu_list: list, lang: str):
    bleu = {f'annotations_{lang}': i, "bleu_score": []}
    for comb in text_combinations:
        b_s = sentence_bleu([comb[0]], comb[1])
        bleu["bleu_score"].append(b_s)
    bleu_list.append(bleu)
    return bleu_list


def calculate_meteor_score(i: int, text_combinations: list, meteor_list: list, lang: str):
    meteor = {f'annotations_{lang}': i, "meteor_score": []}
    for comb in text_combinations:
        m_s = nltk.translate.meteor_score.meteor_score([comb[0]], comb[1])
        meteor["meteor_score"].append(m_s)
    meteor_list.append(meteor)
    return meteor_list


def calculate_bert_score(i: int, text_combinations: list, bert_list: list, lang: str) -> list:
    bert = {f'annotations_{lang}': i, "P": [], "R": [], "F1": []}
    for comb in text_combinations:
        P, R, F1 = bert_score([comb[0]], [comb[1]], lang="other", verbose=True)
        bert["P"].append(P)
        bert["R"].append(R)
        bert["F1"].append(F1)
    # plot_example(text_combinations[0][0], text_combinations[0][1], lang="other")
    bert_list.append(bert)
    return bert_list


def calculate_text_scores(df: pd.DataFrame, lang: str) -> (pd.DataFrame, pd.DataFrame):
    bert_list = []
    meteor_list = []
    rouge_list = []
    bleu_list = []
    for value_list in df[
        [f'annotations_{lang}', f"tokens_id_{PERSONS[0]}", f"tokens_id_{PERSONS[1]}", f"tokens_id_{PERSONS[2]}",
         f'tokens_dict_{PERSONS[0]}', f'tokens_ws_dict_{PERSONS[0]}',
         f'tokens_dict_{PERSONS[1]}', f'tokens_ws_dict_{PERSONS[1]}',
         f'tokens_dict_{PERSONS[2]}', f'tokens_ws_dict_{PERSONS[2]}']].values:
        text_combinations = get_text_combinations(copy.deepcopy(value_list[1:4]), copy.deepcopy(value_list))
        rouge_list = calculate_rouge_score(value_list[0], text_combinations, rouge_list, lang)
        bert_list = calculate_bert_score(value_list[0], text_combinations, bert_list, lang)
        meteor_list = calculate_meteor_score(value_list[0], text_combinations, meteor_list, lang)
        bleu_list = calculate_bleu_score(value_list[0], text_combinations, bleu_list, lang)
    return pd.DataFrame.from_records(rouge_list), pd.DataFrame.from_records(bert_list), pd.DataFrame.from_records(
        meteor_list), pd.DataFrame.from_records(bleu_list)


def get_text(token_list: list, tokens_dict: dict, ws_dict: dict) -> str:
    text = ""
    for nr in token_list:
        if nr != NAN_KEY:
            if ws_dict[int(nr)]:
                text = text + tokens_dict[int(nr)] + " "
            else:
                text = text + tokens_dict[int(nr)]
    return text


def get_text_combinations(token_list: list, token_dict_list: list) -> list:
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


def calculate_IAA(df: pd.DataFrame, lang: str):
    """
    Creates 'normalized_token_dict' column (normalized dictionary for entire row).
    Applies IAA text and numerical scores to preprocessed DataFrame.
    Merges all score columns to DataFrame and applies aggregation min, max, mean to scores.
    Returns DataFrame
    """
    for pers in PERSONS:
        df = preprocessing.normalize_person_tokens(df, pers, lang)
        df = preprocessing.string_to_dict(df, f'tokens_ws_dict_{pers}')
        df = preprocessing.string_to_dict(df, f'tokens_dict_{pers}')

    print("Calculating scores...")
    r, be, m, b = calculate_text_scores(df, lang)
    score_df_list = [calculate_overlap_min_max(df, lang),
                     calculate_jaccard_similarity_distance(df, lang), r, be, m, b]
    for score_df in score_df_list:
        df = df.merge(score_df, on=f'annotations_{"de"}',
                      how='outer')

    for agg in AGGREGATIONS:
        for score in SCORES:
            df = apply_aggregation(df, score, agg)

    return df


def write_IAA_to_csv(df: pd.DataFrame, lang: str, label: str, version: str):
    """
    Calculate IAA_scores of preprocessed DataFrame.
    Writes DataFrame to csv
    """
    df = calculate_IAA(df, lang)
    preprocessing.write_csv(Path("{}/{}_{}.csv".format(lang, f"{label.lower().replace(' ', '_')}_{lang}", version)), df)
    print("Saved {}_{}.csv successfully!".format(f"{label.lower().replace(' ', '_')}_{lang}", version))
