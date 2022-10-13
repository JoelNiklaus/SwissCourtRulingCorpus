"""
Json structure
{
  "id": 48757,
  "year": 2015,
  "facts": "Sachverhalt: A. X._ war bei der Krankenversicherung C._ taggeldversichert. Infolge einer Arbeitsunf\u00e4higkeit leistete ihm die C._ vom 30. Juni 2011 bis am 28. Juni 2013 Krankentaggelder, wobei die Leistungen bis am 30. September 2012 auf Grundlage einer Arbeitsunf\u00e4higkeit von 100% und danach basierend auf einer Arbeitsunf\u00e4higkeit von 55% erbracht wurden. Die Neueinsch\u00e4tzung der Arbeitsf\u00e4higkeit erfolgte anhand eines Gutachtens der D._ AG vom 27. August 2012, welches im Auftrag der C._ erstellt wurde. X._ machte daraufhin gegen\u00fcber der C._ geltend, er sei entgegen dem Gutachten auch nach dem 30. September 2012 zu 100% arbeitsunf\u00e4hig gewesen. Ferner verlangte er von der D._ AG zwecks externer \u00dcberpr\u00fcfung des Gutachtens die Herausgabe s\u00e4mtlicher diesbez\u00fcglicher Notizen, Auswertungen und Unterlagen. A._ (als Gesch\u00e4ftsf\u00fchrer der D._ AG) und B._ (als f\u00fcr das Gutachten medizinisch Verantwortliche) antworteten ihm, dass sie alle Unterlagen der C._ zugestellt h\u00e4tten und dass allf\u00e4llige Fragen zum Gutachten direkt der C._ zu stellen seien. X._ reichte am 2. Januar 2014 eine Strafanzeige gegen A._ und B._ ein. Er wirft diesen vor, ihn durch die Nichtherausgabe der Dokumente und durch Behinderung des IV-Verfahrens gen\u00f6tigt, Daten besch\u00e4digt bzw. vernichtet und ein falsches \u00e4rztliches Zeugnis ausgestellt zu haben. Zudem h\u00e4tten sie durch die Verz\u00f6gerung des IV-Verfahrens und insbesondere durch das falsche \u00e4rztliche Zeugnis sein Verm\u00f6gen arglistig gesch\u00e4digt. B. Die Staatsanwaltschaft des Kantons Bern, Region Oberland, nahm das Verfahren wegen N\u00f6tigung, Datenbesch\u00e4digung, falschem \u00e4rztlichem Zeugnis und arglistiger Verm\u00f6genssch\u00e4digung mit Verf\u00fcgung vom 10. November 2014 nicht an die Hand. Das Obergericht des Kantons Bern wies die von X._ dagegen erhobene Beschwerde am 27. April 2015 ab, soweit darauf einzutreten war. C. X._ beantragt mit Beschwerde in Strafsachen, der Beschluss vom 27. April 2015 sei aufzuheben und die Angelegenheit zur korrekten Ermittlung des Sachverhalts an die Staatsanwaltschaft zur\u00fcckzuweisen. Er stellt zudem den sinngem\u00e4ssen Antrag, das bundesgerichtliche Verfahren sei w\u00e4hrend der Dauer des konnexen Strafverfahrens gegen eine Teilgutachterin und des ebenfalls konnexen Zivil- oder Strafverfahrens gegen die C._ wegen Einsichtsverweigerung in das mutmasslich gef\u00e4lschte Originalgutachten zu sistieren. X._ ersucht um unentgeltliche Rechtspflege. ",
  "labels": 0,  # dismissal
  "language": "de",
  "region": "Espace Mittelland",
  "canton": "be",
  "legal area": "penal law"
}
@Todo Clean this up and and add comments
"""
from pathlib import Path

import pandas as pd

# Constant variable definitions
GOLD_SESSIONS = {"de": "gold_final", "fr": "gold_nina", "it": "gold_nina"}
LABELS = ["Lower court", "Supports judgment", "Opposes judgment", "Neutral"]
CSV_PATH = "../../prodigy_dataset_creation/dataset_scrc/{}/test.csv"
OCCLUSION_COLUMNS = ['id_csv', 'year', 'label', 'language', 'origin_region',
                     'origin_canton', 'legal_area', 'explainability_label', 'occluded_text', 'facts']
NUMBER_OF_EXP = [2,3,4]

NAN_KEY = 10000
ORIGINAL_TEST_SET = pd.DataFrame

from scrc.annotation.judgment_explainability.annotations.preprocessing_functions \
    import LANGUAGES, extract_dataset, get_tokens_dict, extract_values_from_column, get_span_df, group_columns, \
    read_csv, string_to_dict, get_white_space_dicts, write_csv, get_combinations


def process_dataset(datasets: dict, lang: str):
    annotations = datasets[f'annotations_{lang}-{GOLD_SESSIONS[lang]}'][
        datasets[f'annotations_{lang}-{GOLD_SESSIONS[lang]}']["answer"] == "accept"]
    annotations.index.name = f"annotations_{lang}"
    annotations_spans = extract_values_from_column(annotations, "spans", "tokens")
    annotations_tokens = extract_values_from_column(annotations, "tokens", "spans")

    ws_df = annotations.copy().explode("tokens").reset_index()
    ws_df = string_to_dict(
        get_white_space_dicts(ws_df.join(ws_df["tokens"].apply(pd.Series).add_prefix("{}_".format("tokens"))),
                              "id_scrc"), 'tokens_ws_dict')

    for label in LABELS:
        label_df, spans_list = get_span_df(annotations_spans, annotations_tokens, label, lang)
        separated_spans_df = get_separated_label_spans(spans_list, lang)
        separated_spans_tokens_df = process_span_token_df(label_df, separated_spans_df, lang)
        label_df = label_df.drop(['tokens_text', 'tokens_start', 'tokens_end', 'tokens_dict', 'id_csv', 'text',
                                  'tokens_dict'], axis=1).merge(separated_spans_tokens_df, on="id_scrc", how="inner")
        occluded_label_df = process_label_df(label_df, ws_df, label, lang)
        globals()[f"{label.lower().replace(' ', '_')}_{lang}"] = occluded_label_df.drop(["index"], axis=1).to_dict(
            "records")


def process_label_df(l_df: pd.DataFrame, ws_df: pd.DataFrame, label: str,
                     lang: str) -> pd.DataFrame:
    """
    @Todo
    """
    l_df = l_df.merge(ws_df, on="id_scrc", how="inner")
    l_df = occlude_text_1(l_df, label)
    l_df[["explainability_label", "language"]] = label, lang
    l_df = l_df.merge(ORIGINAL_TEST_SET, on="id_csv", how="inner")
    l_df = l_df.append(get_facts(l_df.copy(), lang))[OCCLUSION_COLUMNS]
    assert l_df["year"].between(2017, 2020).all()
    l_df.drop_duplicates(inplace=True)
    return l_df.sort_values(by=["id_csv"]).reset_index().rename(
        columns={'origin_region': 'region', 'origin_canton': 'canton'})


def process_span_token_df(label_df: pd.DataFrame, separated_spans, lang) -> pd.DataFrame:
    """
    @Todo
    """
    columns = [f'annotations_{lang}', 'id_scrc', 'id_csv', 'text', 'tokens_text', 'tokens_id', 'tokens_dict']
    spans_tokens_df = group_columns(get_tokens_dict(label_df, "tokens_id", "tokens_text", "tokens_dict"),
                                    lang)[columns].drop_duplicates()
    spans_tokens_df = splits_string_list(spans_tokens_df)
    spans_tokens_df = separated_spans.join(spans_tokens_df.set_index(f'annotations_{lang}'),
                                           on=f'annotations_{lang}')
    spans_tokens_df.index.name = f'annotations_{lang}'
    return join_span_columns(
        string_to_dict(spans_tokens_df, 'tokens_dict').drop(['tokens_text', 'tokens_id'], axis=1), lang)


def splits_string_list(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits string list column to list column.
    Returns Dataframe.
    """
    df["tokens_id"] = df["tokens_id"].astype(str).str.split(",")
    return df


def get_separated_label_spans(span_list_dict: dict, lang: str) -> pd.DataFrame:
    """
    @Todo
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


def join_span_columns(df: pd.DataFrame, lang) -> pd.DataFrame:
    """
    @Todo
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


def occlude_text_1(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    For each row in Dataframe gets spans as keys for 'tokens_dict' and 'tokens_ws_dict'.
    Creates occlusion_string using the tokens and corresponding whitespaces.
    Removes occlusion_string from "text" row and asserts its removal.
    Appends the occlusion_string and the occluded "text" row to occlusion_list resp. text_list.
    Creates columns from lists, drops "text" columns and returns Dataframe.
    """
    text_list = []
    occlusion_list = []
    for index, row in df.iterrows():
        occlusion_string = ""
        if type(row['spans']) == list:
            for span in row['spans']:
                token = row['tokens_dict'][span]
                if row['tokens_ws_dict'][span]:
                    occlusion_string = occlusion_string + token + " "
                else:
                    occlusion_string = occlusion_string + token
            if label == LABELS[0]:
                row["text"] = row["text"].replace(occlusion_string, "[*]")
            else:
                row["text"] = row["text"].replace(occlusion_string, " ")
            assert row["text"].find(occlusion_string) == -1
        occlusion_list.append(occlusion_string)
        text_list.append(row["text"])
    df[["facts"]] = text_list
    df["occluded_text"] = occlusion_list
    return df.drop(["text"], axis=1)

def occlude_text_n(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Occlusion for dim n @todo
    """
    occlusion_list = []
    text_list = []
    for index, row in df.iterrows():
        occlusion_string = ""
        for number in row['combinations']:
            row["facts"] = row["facts"].replace(row["text_dict"][number], " ")
            occlusion_string = occlusion_string + row["text_dict"][number] + " "
            assert row["facts"].find(row["text_dict"][number]) == -1
        occlusion_list.append(occlusion_string)
        text_list.append(row["facts"])
    df[["facts"]] = text_list
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
    Resets the index, converts to dictionary and writes JSONL using write_JSONL().
    """
    df = pd.DataFrame()
    for label in LABELS[1:]:
        df = df.append(globals()[f"{label.lower().replace(' ', '_')}_{lang}"])
    write_csv(Path(filename), df.reset_index().drop("index", axis=1).rename(columns={"facts": "text",  "id_csv": "id"}))


def permutation(filename: str,lang: str, permutations_number: int):
    df = pd.DataFrame()
    for label in LABELS[1:-1]:
        df_exp = pd.DataFrame.from_records(globals()[f"{label.lower().replace(' ', '_')}_{lang}"])
        df_baseline = df_exp[df_exp["explainability_label"] == "Baseline"]
        df_exp = get_occlusion_string_dict(df_exp[df_exp["explainability_label"] == label][['id_csv', 'occluded_text']].values, permutations_number)

        df_exp = df_baseline.merge(df_exp, on="id_csv", how="inner")
        df_exp = df_exp.explode("combinations")
        df = df.append(occlude_text_n(df_exp, label))
    write_csv(Path(filename), df.reset_index().drop("index", axis=1).rename(columns={"facts": "text","id_csv": "id"}))


def get_occlusion_string_dict(value_pairs: list, permutations_number: int) -> pd.DataFrame:
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
        if len(list(value_pairs_dict[key].keys())) >= permutations_number:
            combinations = get_combinations(list(value_pairs_dict[key].keys()), permutations_number)
            df = df.append({"id_csv": key, "combinations": combinations,
                            "text_dict": value_pairs_dict[key]}, ignore_index=True)
    return df




def insert_lower_courts(filename: str, lang: str):
    lower_court_df = pd.DataFrame.from_records(globals()[f"{LABELS[0].lower().replace(' ', '_')}_{lang}"])
    lower_court_list = list(lower_court_df[lower_court_df["explainability_label"] == LABELS[0]]["occluded_text"])
    normalized_lower_court_list = list(set(normalize_white_spaces(lower_court_list)))
    lower_court_df["lower_court"] = [normalized_lower_court_list] * len(lower_court_df)
    lower_court_df = lower_court_df.explode("lower_court")
    lower_court_df["lower_court"] = lower_court_df.apply(
        lambda row: get_original_court(row["lower_court"], row["occluded_text"]), axis=1)
    lower_court_df["facts"] = lower_court_df.apply(
        lambda row: row["facts"].replace("[*]", row["lower_court"] + " "), axis=1)

    original_court_df = lower_court_df[lower_court_df["lower_court"] == "original court"]
    original_court_df["lower_court"] = original_court_df["occluded_text"]
    original_court_df = original_court_df[["id_csv", "lower_court"]]

    baseline_df = lower_court_df[lower_court_df["explainability_label"] != LABELS[0]].drop("lower_court",
                                                                                           axis=1).drop_duplicates()
    baseline_df = baseline_df.merge(original_court_df, on="id_csv", how="inner").drop("occluded_text", axis=1)

    lower_court_df["occluded_text"] = lower_court_df["lower_court"]
    lower_court_df = lower_court_df[lower_court_df["explainability_label"] == LABELS[0]]
    lower_court_df = lower_court_df[lower_court_df["lower_court"] != "original court"].drop("occluded_text", axis=1)
    write_csv(Path(filename), lower_court_df.append(baseline_df).drop_duplicates().reset_index().drop("index", axis=1).rename(columns={"facts": "text","id_csv": "id"}))


def normalize_white_spaces(string_list: list) -> list:
    normalized_list = []
    for string in string_list:
        if string[-1] == ' ':
            if string[-2] == ' ':
                normalized_list.append(string[:-2])
            else:
                normalized_list.append(string[:-1])
        if string[-1] != ' ':
            normalized_list.append(string)
    return normalized_list


def get_original_court(lower_court, original_court):
    if lower_court in original_court:
        return "original court"
    else:
        return lower_court


if __name__ == '__main__':
    extracted_datasets = extract_dataset("../annotations/{}/gold/gold_annotations_{}.jsonl",
                                         "../annotations/{}/gold/gold_annotations_{}-{}.jsonl")
    for l in LANGUAGES:
        ORIGINAL_TEST_SET = read_csv(CSV_PATH.format(l), "id")[['text', 'label', 'origin_canton', 'origin_region']]
        ORIGINAL_TEST_SET.index.name = "id_csv"
        try:
            process_dataset(extracted_datasets, l)
            appends_df(f"occlusion_test_sets/{l}/occlusion_test_set_{l}_exp_1.csv", l)
            for nr in NUMBER_OF_EXP:
                permutation(f"occlusion_test_sets/{l}/occlusion_test_set_{l}_exp_{nr}.csv",l, nr)
            insert_lower_courts(f"lower_court_test_sets/{l}/lower_court_test_set_{l}.csv", l)
        except KeyError as err:
            print(err)
            pass
