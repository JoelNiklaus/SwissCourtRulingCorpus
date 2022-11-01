from pathlib import Path

import utils.IAA_scores as IAA_scores
import utils.preprocessing as preprocessing
import utils.qualitative_analysis as ql
import utils.quantitative_analysis as qt

LANGUAGES = ["de", "fr", "it"]
PERSONS = ["angela", "lynn", "thomas"]
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
OCCLUSION_PATHS = {"test_sets": ("../occlusion/lower_court_test_sets/{}/lower_court_test_set_{}.csv",
                                 "../occlusion/occlusion_test_sets/{}/occlusion_test_set_{}_exp_{}.csv"),
                   "prediction": ("../occlusion/lower_court_predictions/{}/predictions_test.csv",
                                  "../occlusion/occlusion_predictions/{}_{}/predictions_test.csv")}
NUMBER_OF_EXP = [1, 2, 3, 4]


def occlusion_analysis(datasets: dict, lang: str, version: str):
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
        label_df = preprocessing.get_span_df(annotations_spans, annotations_tokens, label, lang)
        ws_df = preprocessing.get_white_space_dicts(label_df, "annotations_{}".format(lang))
        label_df = preprocessing.get_tokens_dict(label_df, "tokens_id", "tokens_text", "tokens_dict")
        label_df = label_df.join(ws_df[[f"annotations_{lang}", 'tokens_ws_dict']].set_index(f"annotations_{lang}"),
                                 on="annotations_{}".format(lang))
        for pers in PERSONS:
            label_list.append(preprocessing.get_annotator_df(pers, label_df, lang, version))

        label_df = preprocessing.merge_triple(label_list, PERSONS, lang)
        label_df = preprocessing.get_normalize_tokens_dict(label_df)
        if lang == "de":
            IAA_scores.write_IAA_to_csv(label_df, lang, label, version)
        else:
            preprocessing.write_csv(
                Path("../{}/{}_{}.csv".format(lang, f"{label.lower().replace(' ', '_')}_{lang}", version)),
                label_df)
        print("Saved {}_{}.csv successfully!".format(f"{label.lower().replace(' ', '_')}_{lang}", version))



"""

Zuerst explainability wert pro case ausrechnenen (Unterschied zu baseline).
Min max und durchschnittlicher explainability wert

Dann IAA zwischen gs daten set und sätzen. Gleiche schritte wie bei erstllung zuerst 1 satz zu 1 satz dann 2,3,4

- Tokens in occluded text per exp. label and version 
- Gold standard total tokens zählen

"""
if __name__ == '__main__':
    for l in LANGUAGES:
        lower_court_test_set = preprocessing.read_csv(OCCLUSION_PATHS["test_sets"][0].format(l,l), "index")
        lower_court_prediction = preprocessing.temp_scaling(
            preprocessing.read_csv(OCCLUSION_PATHS["prediction"][0].format(l), "id"))
        for nr in NUMBER_OF_EXP:
            occlusion_test_set = preprocessing.read_csv(OCCLUSION_PATHS["test_sets"][1].format(l,l, nr), "index")
            occlusion_prediction = preprocessing.temp_scaling(
                preprocessing.read_csv(OCCLUSION_PATHS["prediction"][1].format(l, nr), "id"))

            print(occlusion_test_set.columns)
