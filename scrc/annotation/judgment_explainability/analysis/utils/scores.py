"""
Sources for the implementations
- Overlap Maximum and Overlap Minimum https://www.geeksforgeeks.org/maximum-number-of-overlapping-intervals/
- [x]
- By looking at the annotated sentences themselves and at the reasoning in the free-text annotation for some of the more complex cases4 a qualitative analysis of
the annotation is also possible.
"""

import copy
import math
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

nltk.download('punkt')
nltk.download('wordnet')
import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing
import rouge_score.rouge_scorer as rs
from bert_score import score as bert_score

LANGUAGES = ["de", "fr", "it"]
PERSONS = ["angela", "lynn", "thomas"]
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
NAN_KEY = 10000
AGGREGATIONS = ["mean", "max", "min"]


def write_IAA_to_csv(df: pd.DataFrame, lang: str, label: str, version: str):
    """
    Calculate IAA_scores of preprocessed DataFrame.
    Writes DataFrame to csv
    @ Todo implement occlusion
    """
    df = calculate_IAA_annotations(df, lang)
    preprocessing.write_csv(Path("{}/{}_{}.csv".format(lang, f"{label.lower().replace(' ', '_')}_{lang}", version)), df)
    print("Saved {}_{}.csv successfully!".format(f"{label.lower().replace(' ', '_')}_{lang}", version))


def calculate_IAA_annotations(df: pd.DataFrame, lang: str) -> pd.DataFrame:
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
    r, be, m, b = calculate_text_scores_annotations(df, lang)
    score_df_list = [calculate_overlap_min_max_annotation(df, lang),
                     calculate_jaccard_similarity_distance_annotation(df, lang), r, be, m, b]
    for score_df in score_df_list:
        df = df.merge(score_df, on=f'annotations_{lang}',
                      how='outer')
    scores = ["overlap_maximum", "overlap_minimum", "jaccard_similarity", "jaccard_distance", "meteor_score",
              "bleu_score"]
    for agg in AGGREGATIONS:
        for score in scores:
            df = apply_aggregation(df, score, agg)

    return df


def calculate_IAA_occlusion(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """
    @Todo Finish functionalities
    Attention id is not a good identifiere since it is the same for a lot of cases!
    """
    print("Calculating scores...")
    df = df.reset_index()
    r, be, m, b = calculate_text_scores_occlusion(df, lang)
    score_df_list = [calculate_overlap_min_max_occlusion(df),
                     calculate_jaccard_similarity_distance_occlusion(df, lang),
                     r.rename(columns={f"annotations_{lang}": "index"}),
                     be.rename(columns={f"annotations_{lang}": "index"}),
                     m.rename(columns={f"annotations_{lang}": "index"}),
                     b.rename(columns={f"annotations_{lang}": "index"})]

    for score_df in score_df_list:
        df = df.merge(score_df, on=f'index',
                      how='outer')
    return df.drop("index", axis=1)


def calculate_text_scores_occlusion(df: pd.DataFrame, lang: str) -> (pd.DataFrame, pd.DataFrame):
    """
    @Todo Comment
    """
    bert_list = []
    meteor_list = []
    rouge_list = []
    bleu_list = []
    for value_list in df[["index", "occluded_text_model", "occluded_text_human"]].values:
        rouge_list = calculate_rouge_score(value_list[0], [(value_list[1], value_list[2])], rouge_list, lang)
        bert_list = calculate_bert_score(value_list[0], [(value_list[1], value_list[2])], bert_list, lang)
        meteor_list = calculate_meteor_score(value_list[0], [(value_list[1], value_list[2])], meteor_list, lang)
        bleu_list = calculate_bleu_score(value_list[0], [(value_list[1], value_list[2])], bleu_list, lang)
    return pd.DataFrame.from_records(rouge_list), pd.DataFrame.from_records(bert_list), pd.DataFrame.from_records(
        meteor_list), pd.DataFrame.from_records(bleu_list)


def calculate_overlap_min_max_occlusion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loops through value_lists, creates dictionary and tokenizes model and human strings.
    Gets length of maximal overlapping sequence for each model vs. human comparison.
    Asserts max length is less than or equal to smallest sample.
    Calculates the overlapping maximum and minimum score using the length of this sequence divided by the maximum or minimum of the sample sets.
    If there is no overlap the overlap_maximum and overlap_minimum equals 0.
    Returns Dataframe containing overlap scores.
    """
    overlap_min_max_list = []
    for value_list in df.copy()[["index", "occluded_text_model", "occluded_text_human"]].values:
        overlap_min_max = {"index": value_list[0], "overlap_maximum": 0,
                           "overlap_minimum": 0}
        tokens_model, tokens_human = word_tokenize(value_list[1]), word_tokenize(value_list[2])
        comb = sorted([tokens_model, tokens_human], key=len)
        len_min_comb, len_max_comb = len(comb[0]), len(comb[1])
        max_overlap = get_max_overlap(comb[0], comb[1])
        if max_overlap != 0:
            assert max_overlap <= len_min_comb
            overlap_min_max["overlap_maximum"] = max_overlap / len_max_comb
            overlap_min_max["overlap_minimum"] = max_overlap / len_min_comb
        overlap_min_max_list.append(overlap_min_max)
    return pd.DataFrame.from_records(overlap_min_max_list)


def get_max_overlap(s1: list, s2: list) -> int:
    """
    Appends continuous overlapping section of two lists of tokens to a list.
    Returns length of maximum overlapping section.
    """
    lst = []
    j = 1
    while j <= len(s1):
        if ''.join(str(i) for i in s1[:j]) in ''.join(str(i) for i in s2):
            lst.append(s1[:j])
            j += 1
        # Section is finished, slice list and check again
        else:
            s1 = s1[j:]
            j = 1
    if len(lst) == 0:
        return 0
    else:
        return max(len(elem) for elem in lst)


def calculate_overlap_min_max_annotation(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """
    Loops through value lists, creates dictionary and gets combinations of the token value_lists.
    For each combination of two lists gets length of maximal overlapping sequence (e.g. [1,2,3] and [2,3,4] -> [2,3]).
    Asserts max length is less than or equal to smallest sample (maximum of overlapping section is section itself).
    Calculates the overlapping maximum and minimum score using the length of this sequence divided by the maximum or minimum of the sample sets.
    If there is no overlap or the sample content is Nan ([10000]) the overlap_maximum and overlap_minimum equals 0.
    Returns Dataframe containing overlap scores.
    """

    overlap_min_max_list = []
    for value_list in df.copy()[
        [f"annotations_{lang}", f"normalized_tokens_{PERSONS[0]}", f"normalized_tokens_{PERSONS[1]}",
         f"normalized_tokens_{PERSONS[2]}", 'normalized_tokens_dict']].values:
        overlap_min_max = {f"annotations_{lang}": value_list[0], "overlap_maximum": [],
                           "overlap_minimum": []}
        combinations = preprocessing.get_combinations(value_list[1:-1], 2)
        for comb in combinations:
            comb = sorted(comb, key=len)
            len_min_comb, len_max_comb = len(comb[0]), len(comb[1])
            max_overlap = get_max_overlap(comb[0], comb[1])
            if max_overlap == 0 or comb == [[NAN_KEY], [NAN_KEY]]:
                overlap_min_max["overlap_maximum"] += [0]
                overlap_min_max["overlap_minimum"] += [0]
            else:
                assert max_overlap <= len_min_comb
                overlap_min_max["overlap_maximum"] += [max_overlap / len_max_comb]
                overlap_min_max["overlap_minimum"] += [max_overlap / len_min_comb]
        overlap_min_max_list.append(overlap_min_max)

    return pd.DataFrame.from_records(overlap_min_max_list)


def calculate_jaccard_similarity_distance_occlusion(df: pd.DataFrame, lang) -> pd.DataFrame:
    """
    Calculates the Jaccard Similarity and Jaccard distance.
    Following this implementation https://pyshark.com/jaccard-similarity-and-jaccard-distance-in-python/
    @Todo
    """
    jaccard_list = []
    for value_list in df.copy()[["index", "occluded_text_model", "occluded_text_human"]].values:
        jaccard = {"index": value_list[0], "jaccard_similarity": 0, "jaccard_distance": 0}
        tokens_model, tokens_human = word_tokenize(value_list[1]), word_tokenize(value_list[2])
        tokens_normalized = preprocessing.normalize_list_length([tokens_model, tokens_human], {"Nan": "Nan"})
        set_1, set_2 = set(list(tokens_normalized[0])), set(list(tokens_normalized[1]))
        # Find intersection of two sets
        nominator_1 = set_1.intersection(set_2)
        # Find symmetric difference of two sets
        nominator_2 = set_1.symmetric_difference(set_2)
        # Find union of two sets
        denominator = set_1.union(set_2)
        # Take the ratio of sizes
        jaccard["jaccard_similarity"] = len(nominator_1) / len(denominator)
        jaccard["jaccard_distance"] = len(nominator_2) / len(denominator)
        jaccard_list.append(jaccard)
    return pd.DataFrame.from_records(jaccard_list)


def calculate_jaccard_similarity_distance_annotation(df: pd.DataFrame, lang) -> pd.DataFrame:
    """
    Calculates the Jaccard Similarity and Jaccard distance.
    Following this implementation https://pyshark.com/jaccard-similarity-and-jaccard-distance-in-python/
    @Todo
    """
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
    """
    Calculates ROUGE-L,ROUGE-Lsum, ROUGE-1, ROUGE-2  originally introduce by Lin, 2004.
    Returns a list containing a dictionary for each row.
    Uses Python ROUGE Implementation via https://pypi.org/project/rouge-score/
    """
    rouge_scores = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    scorer = rs.RougeScorer(rouge_scores, use_stemmer=True)
    rouge = {f'annotations_{lang}': i, rouge_scores[0]: [], rouge_scores[1]: [],
             rouge_scores[2]: [], rouge_scores[3]: []}
    for comb in text_combinations:
        scores = scorer.score(comb[0], comb[1])
        for i in range(len(rouge_scores)):
            rouge[rouge_scores[i]].append(scores[rouge_scores[i]])
    if len(text_combinations) != 0:
        rouge_list.append(rouge)
    return rouge_list


def calculate_bleu_score(i: int, text_combinations: list, bleu_list: list, lang: str):
    """
    Calculates BLEU score (unigram and bigram averaging) originally introduce by Papineni et al., 2001
    Returns a list containing a dictionary for each row.
    Uses nltk.translate.bleu_score.
    """
    bleu = {f'annotations_{lang}': i, "bleu_score": []}
    for comb in text_combinations:
        b_s = sentence_bleu([comb[0]], comb[1])
        bleu["bleu_score"].append(b_s)
    bleu_list.append(bleu)
    return bleu_list


def calculate_meteor_score(i: int, text_combinations: list, meteor_list: list, lang: str):
    """
    Calculates METEOR introduced by Lavie and Agarwal, 2007.
    Returns a list containing a dictionary for each row.
    Uses nltk.translate.meteor_score.meteor_score.
    """
    meteor = {f'annotations_{lang}': i, "meteor_score": []}
    for comb in text_combinations:
        m_s = nltk.translate.meteor_score.meteor_score([comb[0]], comb[1])
        meteor["meteor_score"].append(m_s)
    meteor_list.append(meteor)
    return meteor_list


def calculate_bert_score(i: int, text_combinations: list, bert_list: list, lang: str) -> list:
    """
    Calculates BERTScore originally introduce by Zhang et al., 2020.
    Returns a list containing a dictionary for each row.
    Uses Python BERTScore Implementation via https://pypi.org/project/bert-score/
    @ToDo add plots?
    """
    bert = {f'annotations_{lang}': i, "P": [], "R": [], "F1": []}
    for comb in text_combinations:
        P, R, F1 = bert_score([comb[0]], [comb[1]], lang="other", verbose=True)
        bert["P"].append(P)
        bert["R"].append(R)
        bert["F1"].append(F1)
    # plot_example(text_combinations[0][0], text_combinations[0][1], lang="other")
    bert_list.append(bert)
    return bert_list


def calculate_text_scores_annotations(df: pd.DataFrame, lang: str) -> (pd.DataFrame, pd.DataFrame):
    """
    @todo Comment
    """
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
    """
    @todo Comment
    """
    text = ""
    for nr in token_list:
        if nr != NAN_KEY:
            if ws_dict[int(nr)]:
                text = text + tokens_dict[int(nr)] + " "
            else:
                text = text + tokens_dict[int(nr)]
    return text


def get_text_combinations(token_list: list, token_dict_list: list) -> list:
    """
    @todo Comment
    """
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


def calculate_explainability_score(df: pd.DataFrame):
    """
    Separates baseline entries and non baseline entries.
    Calculates difference between the baseline confidence of a case and the confidence of the occlusion.
    Adds explainability_score as column to Dataframe, appends baseline back to it and returns it.
    """
    score_list = []
    baseline = df[df["explainability_label"] == "Baseline"]
    occlusion = df[df["explainability_label"] != "Baseline"]
    for index, row in occlusion.iterrows():
        baseline_value = baseline[baseline["id"] == row["id"]]["confidence"].max()
        score_list.append(float(baseline_value) - float(row["confidence"]))  # diffrence between baseline and occlusion
    occlusion["explainability_score"] = score_list
    occlusion = occlusion.append(baseline)
    return occlusion


def find_flipped_cases(df: pd.DataFrame):
    """
    Separates baseline entries and non baseline entries.
    Adds boolean column has_flipped (True when baseline prediction equals occlusion prediction).
    Appends baseline back to Dataframe and returns it.
    """
    score_list = []
    baseline = df[df["explainability_label"] == "Baseline"]
    occlusion = df[df["explainability_label"] != "Baseline"]
    for index, row in occlusion.iterrows():
        baseline_value = baseline[baseline["id"] == row["id"]]["prediction"].max()
        if row["prediction"] == baseline_value:
            score_list.append(False)
        else:
            score_list.append(True)
    occlusion["has_flipped"] = score_list
    occlusion = occlusion.append(baseline)
    return occlusion


def get_confidence_direction(df: pd.DataFrame, prediction: int):
    """
    Returns Dataframe with column confidence_direction (-1, 0, 1).
    """
    df["confidence_direction"] = df["explainability_score"].apply(
        lambda row: normalize_exp_score_direction(row, prediction)[0])
    return df


def get_norm_explainability_score(df: pd.DataFrame, prediction: int):
    """
    Returns Dataframe with column norm_explainability_score.
    """
    df["norm_explainability_score"] = df["explainability_score"].apply(
        lambda row: normalize_exp_score_direction(row, prediction)[1])
    return df


def normalize_exp_score_direction(explainability_score: float, prediction: int) -> (int, float):
    """
    Calculates direction of confidence depending on prediction and explainability score (e.g. for prediction 0
    explainability_score <0 means less confidence).
    Normalizes explainability_score (flips sign of explainability_score for prediction 1)
    Returns confidence direction and normalized explainability_score.
    Meaning of return values:
    1: More confident than baseline
    0: equally as confident as baseline
    -1: less confident than baseline
    """
    if explainability_score == 0 or math.isnan(explainability_score):
        return 0, explainability_score
    if prediction == 0:
        if explainability_score < 0:
            return -1, explainability_score
        if explainability_score > 0:
            return 1, explainability_score
    else:
        if explainability_score < 0:
            return 1, abs(explainability_score)
        if explainability_score > 0:
            return -1, (-1) * explainability_score


def lower_court_agg(df: pd.DataFrame) -> (pd.Series, pd.Series, pd.Series, pd.Series):
    """
    @todo Comment, add correct return value or json dump, add Mean of both sum_pos/sum_neg (total mean)
    """
    lower_court_distribution = sort_normal_distribution(df.groupby("lower_court")["id"].count().reset_index()) \
        .reset_index().rename(columns={"index": "lower_court", 0: "count"})
    sum_pos = df[df["confidence_direction"] > 0].groupby("lower_court")[
        ["confidence_direction", "norm_explainability_score"]] \
        .agg({"confidence_direction": "sum", "norm_explainability_score": "mean"}) \
        .reset_index().rename(columns={"norm_explainability_score": "mean_norm_explainability_score"})
    sum_pos = lower_court_distribution.merge(sum_pos, on="lower_court", how="inner")
    sum_pos["confidence_direction"] = sum_pos["confidence_direction"].div(sum_pos["count"].values)
    sum_neg = df[df["confidence_direction"] < 0].groupby("lower_court")[
        ["confidence_direction", "norm_explainability_score"]] \
        .agg({"confidence_direction": "sum", "norm_explainability_score": "mean"}) \
        .reset_index().rename(columns={"norm_explainability_score": "mean_norm_explainability_score"})
    sum_neg = lower_court_distribution.merge(sum_neg, on="lower_court", how="inner")
    sum_neg["confidence_direction"] = sum_neg["confidence_direction"].div(sum_neg["count"].values)
    lower_court_distribution["count"] = lower_court_distribution["count"].div(lower_court_distribution["count"].sum())

    return lower_court_distribution, sum_pos, sum_neg,


def sort_normal_distribution(s: pd.Series)-> pd.DataFrame:
    """
    Sorts according to normal distribution (minimums at beginning and ends).
    Returns sorted Dataframe.
    """
    s_list = s.values.tolist()
    s_dict = {item[0]: item[1:][0] for item in s_list}
    result_list = [len(s_dict) * [None], len(s_dict) * [None]]
    i = 0
    while len(s_dict) > 0:
        result_list[0][-(1 + i)] = min(s_dict.values())
        result_list[1][-(1 + i)] = min(s_dict, key=s_dict.get)
        del s_dict[min(s_dict, key=s_dict.get)]
        result_list[0][i] = min(s_dict.values())
        result_list[1][i] = min(s_dict, key=s_dict.get)
        del s_dict[min(s_dict, key=s_dict.get)]
        i += 1
    return pd.DataFrame(result_list[0], index=result_list[1])


def get_correct_direction(df: pd.DataFrame):
    """
    Adds numeric_label column with values (-1: Supports judgement, 0: Neutral, 1: Opposes judgement).
    Adds correct_direction boolean column (True if numeric_label == confidence direction).
    Splits df into direction sets (0: False, 1: True).
    Returns Dataframe, direction set 0, grouped set 1 and grouped set 0.
    """
    df["numeric_label"] = np.where(df["explainability_label"] == LABELS[1], -1, df["explainability_label"])
    df["numeric_label"] = np.where(df["explainability_label"] == LABELS[2], 1, df["numeric_label"])
    df["numeric_label"] = np.where(df["explainability_label"] == "Neutral", 0, df["numeric_label"])
    df["correct_direction"] = np.where(df["numeric_label"] == df["confidence_direction"], True, False)
    s_0, s_1 = df[df["correct_direction"] == False], df[df["correct_direction"] == True]
    s_1 = s_1.groupby("explainability_label")["correct_direction"].count()
    return df, s_0, s_1, s_0.groupby("explainability_label")["correct_direction"].count()


def occlusion_preprocessing(lang: str, df: pd.DataFrame, filename: str):
    """
    @Todo Comment & clean up
    """
    df = df.set_index("index")
    df = calculate_explainability_score(df)
    df = find_flipped_cases(df)
    df_0, df_1 = df[df["prediction"] == 0].sort_values(by=['explainability_score'],
                                                       ascending=True), \
                 df[df["prediction"] == 1].sort_values(by=['explainability_score'],

                                                       ascending=False)
    df_0, df_1 = get_confidence_direction(df_0, 0), get_confidence_direction(
        df_1, 1)
    df_0, df_1 = get_norm_explainability_score(df_0, 0), get_norm_explainability_score(df_1, 1)
    preprocessing.write_csv(Path(f"{lang}/occlusion/{filename}_0.csv"), df_0)
    preprocessing.write_csv(Path(f"{lang}/occlusion/{filename}_1.csv"), df_1)
    return df_0, df_1
