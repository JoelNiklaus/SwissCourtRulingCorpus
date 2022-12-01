import sys
from pathlib import Path

import pandas as pd

import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing
import scrc.annotation.judgment_explainability.analysis.utils.qualitative_analysis as ql
import scrc.annotation.judgment_explainability.analysis.utils.quantitative_analysis as qt
import scrc.annotation.judgment_explainability.analysis.utils.scores as scores

"""
This script prepares the annotations for the analysis.
Starts the IAA-score quantitative analysis, qualitative analysis workflow.
"""

LANGUAGES = ["de", "fr", "it"]
PERSONS = ["angela", "lynn", "thomas"]
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
LABELS_OCCLUSION = ["Lower court", "Supports judgment", "Opposes judgment", "Neutral"]

ANNOTATION_PATHS = {
    "1": ("../legal_expert_annotations/{}/annotations_{}.jsonl",
          "../legal_expert_annotations/{}/annotations_{}-{}.jsonl"),
    "2": ("../legal_expert_annotations/{}/annotations_{}_inspect.jsonl",
          "../legal_expert_annotations/{}/annotations_{}_inspect-{}.jsonl"),
    "3": ("../legal_expert_annotations/{}/annotations_{}_merged.jsonl",
          "../legal_expert_annotations/{}/annotations_{}_merged-{}.jsonl"),
    "4": ("../legal_expert_annotations/{}/gold/gold_annotations_{}.jsonl",
          "../legal_expert_annotations/{}/gold/gold_annotations_{}-{}.jsonl")
}


def annotation_analysis_gold(lang: str, spans_df: pd.DataFrame, tokens_df: pd.DataFrame, version: str):
    """
    This workflow prepares the prodigy JSNOL for the quantitative analysis (only gold-standard).
    It combines the different annotation to one DataFrame per Label (label_df).
    Saves processed datasets as csv.
    Uses utils.preprocessing
    """
    for label in LABELS_OCCLUSION:
        label_df = preprocessing.get_label_df(lang, spans_df, tokens_df, label)

        annotator_df = preprocessing.get_annotator_df(label_df, lang, annotator="", version=version)

        label_df = label_df[
            [f'annotations_{lang}', 'text', 'facts_length', 'year', 'file_number', 'link', 'header',
             'legal_area', 'judgment', 'is_correct']].drop_duplicates() \
            .merge(annotator_df, on=f'annotations_{lang}', how="inner")

        preprocessing.write_csv(
            Path("{}/{}_{}.csv".format(lang, f"{label.lower().replace(' ', '_')}_{lang}", version)),
            label_df)
        print("Saved {}_{}.csv successfully!".format(f"{label.lower().replace(' ', '_')}_{lang}", version))


def annotation_analysis(lang: str, spans_df: pd.DataFrame, tokens_df: pd.DataFrame, version: str):
    """
    This workflow prepares the prodigy JSNOL for the application of the IAA metrics.
    It combines the different annotation to one DataFrame per Label (label_df).
    Normalizes token dictionaries.
    Applies IAA_scores to German subset and writes them to csv.
    Saves processed Italian and French dataset as csv.
    Uses utils.preprocessing
    """
    for label in LABELS:
        label_list = []
        label_df = preprocessing.get_label_df(lang, spans_df, tokens_df, label)
        for pers in PERSONS:
            label_list.append(preprocessing.get_annotator_df(label_df, lang, pers, version))
        label_df = preprocessing.merge_triple(label_list, PERSONS, lang)
        # Normalizes token dictionary
        label_df = preprocessing.get_normalize_tokens_dict(label_df)
        if lang == "de":
            scores.write_IAA_to_csv(label_df, lang, label, version)
        else:
            preprocessing.write_csv(
                Path("{}/{}_{}.csv".format(lang, f"{label.lower().replace(' ', '_')}_{lang}", version)),
                label_df)
            print("Saved {}_{}.csv successfully!".format(f"{label.lower().replace(' ', '_')}_{lang}", version))


if __name__ == '__main__':
    """
    Argument handling, dataset extraction.
    """
    assert len(sys.argv) == 2
    qt.annotation_analysis()
    language = sys.argv[1]
    for ver in range(1, 5):
        df_1, df_2 = preprocessing.annotation_preprocessing([ANNOTATION_PATHS[str(ver)][0],
                                                             ANNOTATION_PATHS[str(ver)][1]],
                                                            language, f"annotations_{language}")[1:]
        if ver == 4:
            annotation_analysis_gold(language, df_1, df_2, str(ver))
        else:
            annotation_analysis(language, df_1, df_2, str(ver))
    qt.annotation_analysis()
    ql.annotation_analysis()
