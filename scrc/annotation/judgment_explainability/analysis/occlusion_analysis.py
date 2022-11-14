import sys
from pathlib import Path

import pandas as pd

import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing
import scrc.annotation.judgment_explainability.analysis.utils.qualitative_analysis as ql
import scrc.annotation.judgment_explainability.analysis.utils.quantitative_analysis as qt
import scrc.annotation.judgment_explainability.analysis.utils.scores as scores

LANGUAGES = ["de", "fr", "it"]
PERSONS = ["angela", "lynn", "thomas"]
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
OCCLUSION_PATHS = {"test_sets": ("../occlusion/lower_court_test_sets/{}/lower_court_test_set_{}.csv",
                                 "../occlusion/occlusion_test_sets/{}/occlusion_test_set_{}_exp_{}.csv"),
                   "prediction": ("../occlusion/lower_court_predictions/{}/predictions_test.csv",
                                  "../occlusion/occlusion_predictions/{}_{}/predictions_test.csv")}
NUMBER_OF_EXP = [1, 2, 3, 4]


def occlusion_analysis(lang: str, mode: str, exp_nr: int):
    """
    @todo finish documentation, clean up, refactor and finish IAA calculation and dumps
    """
    baseline_test_set = preprocessing.read_csv(OCCLUSION_PATHS["test_sets"][1].format(lang, lang, 1), "id")
    baseline_prediction = preprocessing.temp_scaling(
        preprocessing.read_csv(OCCLUSION_PATHS["prediction"][1].format(lang, 1),
                               "id")).reset_index().reset_index().drop(["label"], axis=1)
    baseline_df = pd.merge(baseline_test_set, baseline_prediction, on="index", suffixes=(f'_test_set', f'_prediction'),
                           how="outer").drop_duplicates()
    baseline_df = baseline_df[baseline_df["explainability_label"] == "Baseline"]

    if mode == "lower_court":
        test_set = preprocessing.read_csv(OCCLUSION_PATHS["test_sets"][0].format(lang, lang), "index").reset_index()
        prediction = preprocessing.temp_scaling(
            preprocessing.read_csv(OCCLUSION_PATHS["prediction"][0].format(lang),
                                   "id")).reset_index().reset_index().drop(["id", "label"], axis=1)
        filename = f"bias_{mode}_{lang}"

    if mode == "occlusion":
        test_set = preprocessing.read_csv(OCCLUSION_PATHS["test_sets"][1].format(lang, lang, exp_nr), "id")
        prediction = preprocessing.temp_scaling(
            preprocessing.read_csv(OCCLUSION_PATHS["prediction"][1].format(lang, exp_nr),
                                   "id")).reset_index().reset_index().drop(["label"], axis=1)
        filename = f"{mode}_{nr}_{lang}"

    occlusion = pd.merge(prediction, test_set, on="index",
                         suffixes=(f'_test_set', f'_prediction'),
                         how="outer").drop_duplicates()
    if exp_nr in NUMBER_OF_EXP[1:]:
        occlusion = occlusion.append(baseline_df)
    occlusion = occlusion.set_index("index")
    occlusion = scores.calculate_explainability_score(occlusion)
    occlusion = scores.find_flipped_cases(occlusion)
    occlusion_0, occlusion_1 = occlusion[occlusion["prediction"] == 0].sort_values(by=['explainability_score'],
                                                                                   ascending=True), \
                               occlusion[occlusion["prediction"] == 1].sort_values(by=['explainability_score'],

                                                                                   ascending=False)
    occlusion_0, occlusion_1 = scores.get_confidence_direction(occlusion_0, 0), scores.get_confidence_direction(
        occlusion_1, 1)
    occlusion_0, occlusion_1 = scores.get_norm_explainability_score(occlusion_0,
                                                                    0), scores.get_norm_explainability_score(
        occlusion_1, 1)

    preprocessing.write_csv(Path(f"{lang}/occlusion/{filename}_0.csv"), occlusion_0)
    preprocessing.write_csv(Path(f"{lang}/occlusion/{filename}_1.csv"), occlusion_1)

    occlusion = occlusion_0.append(occlusion_1).drop_duplicates()
    occlusion = occlusion[occlusion["explainability_label"] != "Baseline"]
    if mode == "lower_court":
        scores.lower_court_agg(occlusion)

    if mode == "occlusion":
        scores.get_correct_direction(occlusion)
        occlusion, false_classification, score_1, score_0 = scores.get_correct_direction(occlusion)
        false_classification = false_classification[false_classification["confidence_direction"] != 0]
        false_classification_pos, false_classification_neg = false_classification[
                                                                 false_classification["confidence_direction"] == 1], \
                                                             false_classification[
                                                                 false_classification["confidence_direction"] == -1]
        o_judgement, s_judgement = occlusion[occlusion["numeric_label"] == 1], occlusion[
            occlusion["numeric_label"] == -1]
        o_judgement, s_judgement = o_judgement[["id", "explainability_label", "numeric_label", "occluded_text"]], \
                                   s_judgement[["id", "explainability_label", "numeric_label", "occluded_text"]]
        o_judgement = pd.merge(false_classification_pos, o_judgement, on="id",
                               suffixes=(f'_model', f'_human'),
                               how="inner").drop_duplicates()
        s_judgement = pd.merge(false_classification_neg, s_judgement, on="id",
                               suffixes=(f'_model', f'_human'),
                               how="inner").drop_duplicates()

        scores.calculate_IAA_occlusion(o_judgement, lang)


if __name__ == '__main__':
    assert len(sys.argv) == 2
    occlusion_analysis(sys.argv[1], "lower_court", 0)
    for nr in NUMBER_OF_EXP:
        occlusion_analysis(sys.argv[1], "occlusion", nr)
    qt.occlusion_analysis()
    ql.occlusion_analysis()
