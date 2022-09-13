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
"""

import ast
import pandas as pd

GOLD_SESSIONS = {"de":"gold_final", "fr":"gold_nina", "it":"gold_nina"}
LABELS = ["Lower court", "Supports judgment", "Opposes judgment", "Neutral"]
CSV_PATHS = {"de":"../../prodigy_dataset_creation/dataset_scrc/de/test.csv" , "fr":"../../prodigy_dataset_creation/dataset_scrc/fr/test.csv", "it":"../../prodigy_dataset_creation/dataset_scrc/it/test.csv"}
from scrc.annotation.judgment_explainability.annotations.analysis.preprocessing_functions \
    import LANGUAGES, extract_dataset,get_tokens_dict ,extract_values_from_column, get_span_df, group_columns


from scrc.annotation.prodigy_dataset_creation.dataset_creation_functions import read_csv
# Sets pandas print options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def process_dataset(datasets: dict, lang: str):
    dataset_information = read_csv(CSV_PATHS[lang])[['text','label','origin_canton','origin_region']]
    dataset_information.index.name = "id_csv"
    annotations = datasets[f'annotations_{lang}-{GOLD_SESSIONS[lang]}'][
        datasets[f'annotations_{lang}-{GOLD_SESSIONS[lang]}']["answer"] == "accept"]
    annotations.index.name = f"annotations_{lang}"
    annotations_spans = extract_values_from_column(annotations, "spans", "tokens")
    annotations_tokens = extract_values_from_column(annotations, "tokens", "spans")
    tokens_ws = transform_tokens_dict(get_white_space_dicts(annotations),'tokens_ws_dict')
    for label in LABELS:
        label_df = get_span_df(annotations_spans, annotations_tokens, label, lang)[0]
        spans_tokens_df = get_span_token_df(get_tokens_dict(label_df,"tokens_id","tokens_text", "tokens_dict"), lang)
        separated_spans_df = get_separated_label_spans(get_span_df(annotations_spans, annotations_tokens, label, lang)[1],lang)
        spans_tokens_df = separated_spans_df.join(spans_tokens_df.set_index(f'annotations_{lang}'), on=f'annotations_{lang}')
        spans_tokens_df.index.name = f'annotations_{lang}'

        spans_tokens_df = transform_tokens_dict(spans_tokens_df, 'tokens_dict').drop(['tokens_text', 'tokens_id'], axis=1)
        spans_tokens_df = join_span_columns(spans_tokens_df, lang)
        label_df = label_df.drop(['tokens_text', 'tokens_start', 'tokens_end','tokens_dict', 'id_csv', 'text',
       'tokens_dict'], axis=1).merge(spans_tokens_df,on="id_scrc", how="inner")
        label_df = label_df.merge(tokens_ws, on="id_scrc", how="inner")
        label_df = occlude_text(label_df)
        label_df["explainability_label"] = label
        label_df["language"] = label

        label_df = label_df.merge(dataset_information,on="id_csv", how="inner")
        original_facts_row = label_df.copy()
        original_facts_row = original_facts_row.drop_duplicates(subset=["id_csv"], inplace=False, keep='first').drop("facts", axis=1).rename(columns={"text": "facts"})
        label_df = label_df.append(original_facts_row)
        label_df = label_df[['id_csv', 'year','facts','label','language','origin_region','origin_canton','legal_area',
                                                                           'explainability_label']]

        label_df.drop_duplicates(inplace=True)
        label_df = label_df.sort_values(by=["id_csv"]).reset_index().rename(columns={'origin_region':'region','origin_canton':'canton'})

        globals()[f"{label.lower().replace(' ', '_')}_{lang}"] = label_df.drop(["index"], axis=1)

        print(globals()[f"{label.lower().replace(' ', '_')}_{lang}"]["facts"].values)


def get_span_token_df(tokens:pd.DataFrame, lang) -> pd.DataFrame:
    tokens = group_columns(tokens, lang)
    tokens = tokens[[f'annotations_{lang}', 'id_scrc', 'id_csv','text','tokens_text', 'tokens_id', 'tokens_dict']]
    tokens = tokens.drop_duplicates()
    tokens["tokens_id"] = tokens["tokens_id"].astype(str).str.split(",")
    return tokens

def get_separated_label_spans(span_list_dict: dict, lang: str)-> pd.DataFrame:
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

def transform_tokens_dict(df: pd.DataFrame, col_name) -> pd.DataFrame:
    tokens_dict_list = []
    for token_dict in df[col_name].values:
        if type(token_dict) == str:
            token_dict = ast.literal_eval(token_dict)
        tokens_dict_list.append(token_dict)
    df[col_name] = tokens_dict_list
    return df

def join_span_columns(df: pd.DataFrame, lang) -> pd.DataFrame:
    df_list = []
    for col in df.columns:
        if col.startswith("span_"):
            span_cols = df[["id_scrc", "id_csv", "text", "tokens_dict", col]].dropna().rename(
                columns={col: "spans"})
            df_list.append(span_cols)

    df_separated = df_list[0]
    for i in range(1, len(df_list)):
        df_separated = pd.concat([df_separated, df_list[i]])

    return  df_separated.reset_index().drop(f'annotations_{lang}', axis=1)

def get_white_space_dicts(df: pd.DataFrame)-> pd.DataFrame:
    ws = df.explode("tokens").reset_index()
    df_ws =  ws["tokens"].apply(pd.Series).add_prefix("{}_".format("tokens"))
    ws = ws.join(df_ws)
    ws = get_tokens_dict(ws, 'tokens_id', 'tokens_ws', 'id_ws_dict')[['id_scrc','id_ws_dict']]
    ws['tokens_ws_dict'] = ws.groupby(['id_scrc'])['id_ws_dict'].transform(
        lambda x: "{{{}}}".format(','.join(x.astype(str)).replace("{","").replace("}","")))
    return ws.drop('id_ws_dict', axis=1).drop_duplicates()

def occlude_text(df: pd.DataFrame) -> pd.DataFrame:
    text_list = []
    for index, row in df.iterrows():
        occlusion_string =""
        if type(row['spans']) == list:
            for span in row['spans']:
                token = row['tokens_dict'][span]
                if row['tokens_ws_dict'][span]:
                    occlusion_string = occlusion_string + token + " "
                else:
                    occlusion_string = occlusion_string + token
            row["text"] = row["text"].replace(occlusion_string, "[tokens removed] ")
            assert row["text"].find("[tokens removed]")!=-1
        text_list.append(row["text"])
    df["facts"] = text_list
    return df.drop(["text"], axis=1)


if __name__ == '__main__':
    extracted_datasets = extract_dataset("../annotations/{}/gold/gold_annotations_{}.jsonl", "../annotations/{}/gold/gold_annotations_{}-{}.jsonl")
    for l in LANGUAGES:
        try:
            process_dataset(extracted_datasets, l)
        except KeyError as err:
            print(err)
            pass














