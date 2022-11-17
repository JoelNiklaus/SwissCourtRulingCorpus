import sys
from pathlib import Path

import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing
import scrc.annotation.judgment_explainability.analysis.utils.qualitative_analysis as ql
import scrc.annotation.judgment_explainability.analysis.utils.quantitative_analysis as qt
import scrc.annotation.judgment_explainability.analysis.utils.scores as IAA_scores

LANGUAGES = ["de", "fr", "it"]
PERSONS = ["angela", "lynn", "thomas"]
PERSON_NUMBER = {"angela": 1, "lynn": 2, "thomas": 3}
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
ANNOTATION_PATHS = {
    "1": ("../legal_expert_annotations/{}/annotations_{}.jsonl",
          "../legal_expert_annotations/{}/annotations_{}-{}.jsonl"),
    "2": ("../legal_expert_annotations/{}/annotations_{}_inspect.jsonl",
          "../legal_expert_annotations/{}/annotations_{}_inspect-{}.jsonl"),
    "3": ("../legal_expert_annotations/{}/annotations_{}_merged.jsonl",
          "../legal_expert_annotations/{}/annotations_{}_merged-{}.jsonl")}


def annotation_analysis(datasets: dict, lang: str, version: str):
    """
    This workflow prepares the prodigy JSNOL for the application of the IAA metrics.
    It combines the different annotation to one DataFrame per Label (label_df).
    Normalizes token dictionaries.
    Applies IAA_scores to German subset and writes them to csv.
    Saves processed Italian and French dataset as csv.
    Uses utils.preprocessing

    """
    annotations = datasets["annotations_{}".format(lang)][
        datasets["annotations_{}".format(lang)]["answer"] == "accept"]
    annotations_spans = preprocessing.extract_values_from_column(annotations, "spans", "tokens")
    annotations_tokens = preprocessing.extract_values_from_column(annotations, "tokens", "spans")
    for label in LABELS:
        label_list = []
        label_df = preprocessing.get_span_df(annotations_spans, annotations_tokens, label, lang)[0]
        # Getting the necessary dictionaries
        ws_df = preprocessing.get_white_space_dicts(label_df, "annotations_{}".format(lang))
        label_df = preprocessing.join_to_dict(label_df, "tokens_id", "tokens_text", "tokens_dict")
        label_df = label_df.join(ws_df[[f"annotations_{lang}", 'tokens_ws_dict']].set_index(f"annotations_{lang}"),
                                 on="annotations_{}".format(lang))
        for pers in PERSONS:
            label_list.append(preprocessing.get_annotator_df(pers, label_df, lang, version))

        label_df = preprocessing.merge_triple(label_list, PERSONS, lang)
        # Normalizes token dictionary
        label_df = preprocessing.get_normalize_tokens_dict(label_df)
        if lang == "de":
            IAA_scores.write_IAA_to_csv(label_df, lang, label, version)
        else:
            preprocessing.write_csv(
                Path("{}/{}_{}.csv".format(lang, f"{label.lower().replace(' ', '_')}_{lang}", version)),
                label_df)
            print("Saved {}_{}.csv successfully!".format(f"{label.lower().replace(' ', '_')}_{lang}", version))


if __name__ == '__main__':
    assert len(sys.argv) == 2
    l = sys.argv[1]
    for ver in range(1, 4):
        annotation_analysis(preprocessing.extract_dataset(ANNOTATION_PATHS[str(ver)][0], ANNOTATION_PATHS[str(ver)][1]),
                            l,
                            str(ver))

    qt.annotation_analysis()
    ql.annotation_analysis()
