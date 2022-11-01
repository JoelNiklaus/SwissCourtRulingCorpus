from pathlib import Path

import utils.IAA_scores as IAA_scores
import utils.preprocessing as preprocessing
import utils.qualitative_analysis as ql
import utils.quantitative_analysis as qt

LANGUAGES = ["de", "fr", "it"]
PERSONS = ["angela", "lynn", "thomas"]
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
ANNOTATION_PATHS = [
    ("../legal_expert_annotations/{}/annotations_{}.jsonl", "../legal_expert_annotations/{}/annotations_{}-{}.jsonl"),
    ("../legal_expert_annotations/{}/annotations_{}_inspect.jsonl",
     "../legal_expert_annotations/{}/annotations_{}_inspect-{}.jsonl"),
    ("../legal_expert_annotations/{}/annotations_{}_merged.jsonl",
     "../legal_expert_annotations/{}/annotations_{}_merged-{}.jsonl")]


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
        label_df = preprocessing.get_span_df(annotations_spans, annotations_tokens, label, lang)
        # Getting the necessary dictionaries
        ws_df = preprocessing.get_white_space_dicts(label_df, "annotations_{}".format(lang))
        label_df = preprocessing.get_tokens_dict(label_df, "tokens_id", "tokens_text", "tokens_dict")
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
                Path("../{}/{}_{}.csv".format(lang, f"{label.lower().replace(' ', '_')}_{lang}", version)),
                label_df)
            print("Saved {}_{}.csv successfully!".format(f"{label.lower().replace(' ', '_')}_{lang}", version))


if __name__ == '__main__':
    for paths in ANNOTATION_PATHS:
        extracted_datasets = preprocessing.extract_dataset(paths[0], paths[1])
        for l in LANGUAGES:
            for ver in ["1","2","3"]:
                annotation_analysis(extracted_datasets, l, ver)
    qt.annotation_analysis()
    ql.analysis()
