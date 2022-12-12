from pathlib import Path

import pandas as pd

import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing

"""
This file creates the test sets for the occlusion and lower_court experiments.
The following columns are mandatory to provide the correct format for the csv:
 ["index","id","year","label","language","region","canton","legal area","explainability_label","occluded_text","text"]
"""

TEST_SET_PATH = "../../prodigy_dataset_creation/dataset_scrc/{}/test.csv"
GOLD_ANNOTATIONS = ["../legal_expert_annotations/{}/gold/gold_annotations_{}.jsonl",
                    "../legal_expert_annotations/{}/gold/gold_annotations_{}-{}.jsonl"]
OCCLUSION_COLUMNS = ['id_csv', 'year', 'label', 'language', 'origin_region',
                     'origin_canton', 'legal_area', 'explainability_label', 'occluded_text', 'facts']

ORIGINAL_TEST_SET = pd.DataFrame

LANGUAGES = ["de", "fr", "it"]
GOLD_SESSION = "gold_nina"
PERSONS = ["angela", "lynn", "thomas"]
LABELS_OCCLUSION = ["Lower court", "Supports judgment", "Opposes judgment", "Neutral"]
NUMBER_OF_EXP = [2, 3, 4]


def process_dataset(annotation_df: pd.DataFrame, spans_df: pd.DataFrame, tokens_df: pd.DataFrame, lang: str):
    """
    Creates whitespace Dataframe, gets label_df.
    For each court case separates annotations into sentences/continuous annotations.
    Starts occlusion process.
    Assigns global vars.
    """

    ws_df = annotation_df.copy().explode("tokens").reset_index()
    ws_df = preprocessing.string_to_dict(
        preprocessing.get_white_space_dicts(
            ws_df.join(ws_df["tokens"].apply(pd.Series).add_prefix("{}_".format("tokens"))),
            "id_scrc"), 'tokens_ws_dict')

    for label in LABELS_OCCLUSION:
        label_df, spans_list = preprocessing.get_span_df(spans_df, tokens_df, label, lang)
        separated_spans_df = get_separated_label_spans(spans_list, lang)
        separated_spans_tokens_df = process_span_token_df(label_df, separated_spans_df, lang)
        label_df = label_df.drop(['tokens_text', 'tokens_start', 'tokens_end', 'tokens_dict', 'id_csv', 'text',
                                  ], axis=1).merge(separated_spans_tokens_df, on="id_scrc", how="inner")
        occluded_label_df = process_label_df(label_df, ws_df, label, lang)
        globals()[f"{label.lower().replace(' ', '_')}_{lang}"] = occluded_label_df.drop(["index"], axis=1).to_dict(
            "records")


def get_separated_label_spans(span_list_dict: dict, lang: str) -> pd.DataFrame:
    """
    Creates dict span_dict of shape {index: [[token_ids_span_1],[token_ids_span_2]]}.
    Returns Dataframe with each column having the token_ids of the different spans.
    """
    i = 1
    j = 0
    span_dict = {}
    keys = list(span_list_dict.keys())
    span_list_index = keys[j].split(".")[0]
    while j < len(span_list_dict.keys()) and i < len(span_list_dict.keys()):
        if span_list_index not in span_dict:
            span_dict[span_list_index] = []
        if span_list_index != keys[i].split(".")[0]:
            span_dict[span_list_index].append(span_list_dict[keys[j]])
        if span_list_index == keys[i].split(".")[0]:
            span_dict[span_list_index].append(span_list_dict[keys[j]])
        j += 1
        i += 1
        span_list_index = keys[j].split(".")[0]

    span_dict_df = pd.DataFrame.from_dict(span_dict, orient='index')
    span_dict_df.columns = [f'span_{col_name}' for col_name in span_dict_df.columns]
    span_dict_df.index.name = f'annotations_{lang}'
    span_dict_df.index = span_dict_df.index.astype(int)
    return span_dict_df


def process_span_token_df(label_df: pd.DataFrame, separated_spans: pd.DataFrame, lang: str) -> pd.DataFrame:
    """
    Creates token_id, token_text dictionary column.
    Joins span_token_df with separated_spans.
    Return Dataframe where each span has its own row.
    """
    columns = [f'annotations_{lang}', 'id_scrc', 'id_csv', 'text', 'tokens_text', 'tokens_id', 'tokens_dict']
    spans_tokens_df = \
        preprocessing.group_token_columns(
            preprocessing.join_to_dict(label_df, "tokens_id", "tokens_text", "tokens_dict"),
            lang)[columns].drop_duplicates()
    spans_tokens_df = splits_string_list(spans_tokens_df)
    spans_tokens_df = separated_spans.join(spans_tokens_df.set_index(f'annotations_{lang}'),
                                           on=f'annotations_{lang}')
    spans_tokens_df.index.name = f'annotations_{lang}'
    return join_span_columns(
        preprocessing.string_to_dict(spans_tokens_df, 'tokens_dict').drop(['tokens_text', 'tokens_id'], axis=1), lang)


def splits_string_list(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits string list column to list column.
    Returns Dataframe.
    """
    df["tokens_id"] = df["tokens_id"].astype(str).str.split(",")
    return df


def join_span_columns(df: pd.DataFrame, lang) -> pd.DataFrame:
    """
    Joins span columns to Dataframe where each separate span has its own row.
    Returns Dataframe.
    """
    df_list = []
    for col in df.columns:
        if col.startswith("span_"):
            span_cols = df[["id_scrc", "id_csv", "text", "tokens_dict", col]].dropna().rename(
                columns={col: "spans"})
            df_list.append(span_cols)

    df_separated = df_list[0]
    for i in range(1, len(df_list)):
        df_separated = pd.concat([df_separated, df_list[i]])

    return df_separated.reset_index().drop(f'annotations_{lang}', axis=1)


def process_label_df(label_df: pd.DataFrame, ws_df: pd.DataFrame, label: str,
                     lang: str) -> pd.DataFrame:
    """
    Merges label_df with ws_df.
    Occludes individual annotations (occluding 1 sentence) and adds label explainability_label and language to each row.
    Merges with ORIGINAL_TEST_SET to get additional row.
    Appends baseline rows (facts with no occlusion).
    Assert that only testset cases are in Dataframe.
    Returns sorted Dataframe
    """
    label_df = label_df.merge(ws_df, on="id_scrc", how="inner")
    label_df = occlude_text_1(label_df, label)
    label_df[["explainability_label", "language"]] = label, lang
    label_df = label_df.merge(ORIGINAL_TEST_SET, on="id_csv", how="inner")
    label_df = label_df.append(get_facts(label_df.copy(), lang))[OCCLUSION_COLUMNS]
    assert label_df["year"].between(2017, 2020).all()
    label_df.drop_duplicates(inplace=True)
    return label_df.sort_values(by=["id_csv"]).reset_index().rename(
        columns={'origin_region': 'region', 'origin_canton': 'canton'})


def occlude_text_1(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    For each row in Dataframe gets spans as keys for 'tokens_dict' and 'tokens_ws_dict'.
    Creates occlusion_string using the tokens and corresponding whitespaces.
    Removes occlusion_string from "text" row and asserts its removal.
    Appends the occlusion_string and the occluded "text" row to occlusion_list resp. text_list.
    Creates columns from lists, drops "text" columns.
    Returns Dataframe.
    """
    text_list = []
    occlusion_list = []
    for index, row in df.iterrows():
        occlusion_string = ""
        if type(row['spans']) == list:
            for span in row['spans']:
                token = row['tokens_dict'][span]
                if row['tokens_ws_dict'][span]:
                    occlusion_string = str(occlusion_string + token + " ")
                else:
                    occlusion_string = str(occlusion_string + token)
            if label == LABELS_OCCLUSION[0]:
                row["text"] = row["text"].replace(occlusion_string, "[*]")
            else:
                row["text"] = row["text"].replace(occlusion_string, " ")
            assert row["text"].find(occlusion_string) == -1
        occlusion_list.append(occlusion_string)
        text_list.append(row["text"])
    df["facts"] = text_list
    df["occluded_text"] = occlusion_list
    return df.drop(["text"], axis=1)


def occlude_text_n(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Occludes sentences according to keys in combination and texts in text_dict (e.g. each combination of
    3 sentences in a given case for a given explainability label)
    Returns Dataframe with occluded facts.
    """
    occlusion_list = []
    text_list = []
    for index, row in df.iterrows():
        occlusion_string = ""
        for number in row['combinations']:
            row["facts"] = row["facts"].replace(row["text_dict"][number], " ")
            occlusion_string = str(occlusion_string + row["text_dict"][number] + " ")
            assert row["facts"].find(row["text_dict"][number]) == -1
        occlusion_list.append(occlusion_string)
        text_list.append(row["facts"])
    df["facts"] = text_list
    df["occluded_text"] = occlusion_list
    df["explainability_label"] = label
    return df


def get_facts(df: pd.DataFrame, lang) -> pd.DataFrame:
    """
    Drops all duplicates according to "id_csv" column (keeps one row for each case).
    Drops the occluded texts ("facts"), renames non occluded "text" to facts.
    Adds "explainability_label", "occluded_text" and "language" columns.
    Returns Dataframe
    """
    facts_df = df.drop_duplicates(subset=["id_csv"], inplace=False, keep='first').drop(
        "facts", axis=1).rename(columns={"text": "facts"})
    facts_df[["explainability_label", "occluded_text", "language"]] = "Baseline", "None", lang
    return facts_df


def appends_df(filename: str, lang: str):
    """
    Appends Dataframes for each expandability label to a Dataframe.
    Resets the index, converts to dictionary
    Dumps first occlusion test set.
    """
    df = pd.DataFrame()
    for label in LABELS_OCCLUSION[1:]:
        df = df.append(globals()[f"{label.lower().replace(' ', '_')}_{lang}"])
    preprocessing.write_csv(filename,
                            df.rename(columns={"facts": "text", "id_csv": "id"}).drop_duplicates().reset_index().drop(
                                "index", axis=1))


def create_lower_court_experiment(filename: str, lang: str):
    """
    Gets Dataframe with explainability_label == Lower_court.
    Creates List of lower courts.
    Normalizes list (removes whitespaces and duplicates).
    Creates new column "lower court" were each case gets assigned each lower court.
    Normalizes original court, gets original court, inserts different lower courts in fact section.
    Separates original court entries and joins w/ baseline entries.
    Joins lower_court_df and baseline_df.
    Saves csv.
    """
    lower_court_df = pd.DataFrame.from_records(globals()[f"{LABELS_OCCLUSION[0].lower().replace(' ', '_')}_{lang}"])
    lower_court_list = list(
        lower_court_df[lower_court_df["explainability_label"] == LABELS_OCCLUSION[0]]["occluded_text"])
    normalized_lower_court_list = list(set(normalize_white_spaces(lower_court_list)))
    lower_court_df["lower_court"] = [normalized_lower_court_list] * len(lower_court_df)
    lower_court_df = preprocessing.apply_functions_lower_court(lower_court_df.explode("lower_court"))

    original_court_df = assign_original_court(lower_court_df)
    baseline_df = get_lower_court_baseline(lower_court_df, original_court_df)

    lower_court_df["occluded_text"] = lower_court_df["lower_court"]
    lower_court_df = lower_court_df[lower_court_df["explainability_label"] == LABELS_OCCLUSION[0]]
    lower_court_df = lower_court_df[lower_court_df["lower_court"] != "original court"].drop("occluded_text", axis=1)
    preprocessing.write_csv(filename,
                            lower_court_df.append(baseline_df).drop_duplicates().reset_index().drop("index",
                                                                                                    axis=1).rename(
                                columns={"facts": "text", "id_csv": "id"}))


def assign_original_court(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns the correct lower court as original court
    Returns Dataframe.
    """
    original_court_df = df[df["lower_court"] == "original court"]
    original_court_df.loc[:, "lower_court"] = original_court_df["occluded_text"]
    return original_court_df[["id_csv", "lower_court"]]


def get_lower_court_baseline(lower_court_df: pd.DataFrame, original_court_df: pd.DataFrame):
    """
    Gets baseline rows from lower_court_df merges with lower_court column of original_court_df.
    Returns Dataframe.
    """
    baseline_df = lower_court_df[lower_court_df["explainability_label"] == "Baseline"].drop("lower_court",
                                                                                            axis=1).drop_duplicates()
    return baseline_df.merge(original_court_df, on="id_csv", how="inner").drop("occluded_text", axis=1)


def permutation(filename: str, lang: str, permutations_number: int):
    """
    Gets Dataframe for each explainability label and baseline
    Gets sentence combinations and dict containing all occluded sentnences.
    Merges occlusion_df with baseline_df and explodes "combination" column.
    Occludes the different sentence combinations.
    Saves csv where each row is a different occlusion combination.
    """
    df = pd.DataFrame()
    for label in LABELS_OCCLUSION[1:]:
        occlusion_df = pd.DataFrame.from_records(globals()[f"{label.lower().replace(' ', '_')}_{lang}"])
        baseline_df = occlusion_df[occlusion_df["explainability_label"] == "Baseline"]
        occlusion_df = get_occlusion_string_dict(
            occlusion_df[occlusion_df["explainability_label"] == label][['id_csv', 'occluded_text']].values,
            permutations_number)

        occlusion_df = baseline_df.merge(occlusion_df, on="id_csv", how="inner")
        occlusion_df = occlusion_df.explode("combinations")
        df = df.append(occlude_text_n(occlusion_df, label))
    preprocessing.write_csv(filename,
                            df.reset_index().drop("index", axis=1).rename(columns={"facts": "text", "id_csv": "id"}))


def get_occlusion_string_dict(value_pairs: list, permutation_number: int) -> pd.DataFrame:
    """
    Creates a dict of shape {id_scrc: {0: "first occluded sentence", 1: "second occluded sentence", 2: ...} for number
    of occluded sentence per case and explainability label.
    Gets combinations of sentences according to permutation_number (e.g. for permutation_number = 3, gets each 3 sentence
    combination for the case and the given explainability label.
    Returns Dataframe with list of combinations and text_dict as columns.
    """
    value_pairs_dict = {}
    count = 0
    for pair in value_pairs:
        key, value = pair[0], pair[1]
        if key in value_pairs_dict:
            value_pairs_dict[key][count] = value
            count += 1
        else:
            count = 0
            value_pairs_dict[key] = {count: value}
            count += 1
    df = pd.DataFrame(columns=["id_csv", "combinations", "text_dict"])
    for key in value_pairs_dict.copy():
        if len(list(value_pairs_dict[key].keys())) >= permutation_number:
            combinations = preprocessing.get_combinations(list(value_pairs_dict[key].keys()), permutation_number)
            df = df.append({"id_csv": key, "combinations": combinations,
                            "text_dict": value_pairs_dict[key]}, ignore_index=True)
    return df


def normalize_white_spaces(string_list: list) -> list:
    """
    Checks if there are trailing whitespaces at the end of a lower court and removes them.
    Returns normalized list.
    """
    normalized_list = []
    for string in string_list:
        if string[-1] == ' ':
            if string[-2] == ' ':
                normalized_list.append(string[:-2])
            else:
                normalized_list.append(string[:-1])
        if string[-1] != ' ':
            normalized_list.append(string)
    return list(dict.fromkeys(normalized_list))


if __name__ == '__main__':
    """
    Extracts datasets and starts occlusion and lower_court test set creation
    """
    for l in LANGUAGES:
        df_1, df_2, df_3 = preprocessing.annotation_preprocessing(GOLD_ANNOTATIONS, l,
                                                                  f'annotations_{l}-{GOLD_SESSION}')
        ORIGINAL_TEST_SET = preprocessing.read_csv(TEST_SET_PATH.format(l), "id")[
            ['text', 'label', 'origin_canton', 'origin_region']]
        ORIGINAL_TEST_SET.index.name = "id_csv"
        process_dataset(df_1, df_2, df_3, l)
        appends_df(f"occlusion_test_sets/{l}/occlusion_test_set_{l}_exp_1.csv", l)
        create_lower_court_experiment(f"lower_court_test_sets/{l}/lower_court_test_set_{l}.csv", l)
        for nr in NUMBER_OF_EXP:
            permutation(f"occlusion_test_sets/{l}/occlusion_test_set_{l}_exp_{nr}.csv", l, nr)
