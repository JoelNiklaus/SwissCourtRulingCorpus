import sys

import pandas as pd

import scrc.annotation.judgment_explainability.analysis.utils.plots as plots
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


def lower_court_analysis(lang: str, lower_court_df: pd.DataFrame, filename: str):
    """
    @ToDo documentation scores_lower_court_{lang}
     Label gets cut of T-Test? is difference significant? Wie viele insgesamt?
    """
    lower_court_df_0, lower_court_df_0 = scores.occlusion_preprocessing(lang, lower_court_df, filename)
    lower_court_df = lower_court_df_0.append(lower_court_df_0).drop_duplicates()
    lower_court_df = lower_court_df[lower_court_df["explainability_label"] != "Baseline"]

    lower_court_distribution, sum_pos, sum_neg = scores.lower_court_agg(lower_court_df)
    plots.distribution_plot(lang, lower_court_distribution, col_x="lower_court", col_y="count",
                            label_texts=["Lower Court", "Occurrence of Lower Courts in Dataset"],
                            title=f"Lower Court distribution in the {lang} Dataset")
    plots.lower_court_effect_plot(lang, sum_pos, sum_neg, col_x="lower_court", col_y="confidence_direction",
                                  label_texts=["Confidence Direction", "Lower Court"],
                                  legend_texts=["Positive Influence", "Negativ Influence"],
                                  title="Effect of Lower Courts on the Prediction Confidence", filepath=f'{lang}/plots/lower_court_effect.png')
    plots.lower_court_effect_plot(lang, sum_pos, sum_neg, col_x="lower_court", col_y="mean_norm_explainability_score",
                                  label_texts=["Mean explainability score for one direction", "Lower Court"],
                                  legend_texts=["Exp_Score > 0", "Exp_Score < 0"],
                                  title="Mean distribution on Explainability Scores in both directions", filepath=f'{lang}/plots/mean_explainability_score.png')


def occlusion_analysis(lang: str, occlusion_df: pd.DataFrame, filename: str, exp_nr: int, baseline_df):
    """
    @todo finish documentation, clean up, refactor and finish IAA calculation and dumps,
    """
    if exp_nr in NUMBER_OF_EXP[1:]:
        occlusion_df = occlusion_df.append(baseline_df)
    occlusion_df_0, occlusion_df_0 = scores.occlusion_preprocessing(lang, occlusion_df, filename)

    occlusion_df = occlusion_df_0.append(occlusion_df_0).drop_duplicates()
    occlusion_df = occlusion_df[occlusion_df["explainability_label"] != "Baseline"]
    occlusion_df, false_classification, score_1, score_0 = scores.get_correct_direction(occlusion_df)
    false_classification = false_classification[
        false_classification["confidence_direction"] != 0]  # ignore neutral classification
    false_classification_pos, false_classification_neg = false_classification[
                                                             false_classification["confidence_direction"] == 1], \
                                                         false_classification[
                                                             false_classification["confidence_direction"] == -1]
    o_judgement, s_judgement = occlusion_df[occlusion_df["numeric_label"] == 1], occlusion_df[
        occlusion_df["numeric_label"] == -1]
    o_judgement, s_judgement = o_judgement[["id", "explainability_label", "numeric_label", "occluded_text"]], \
                               s_judgement[["id", "explainability_label", "numeric_label", "occluded_text"]]
    o_judgement, s_judgement = pd.merge(false_classification_pos, o_judgement, on="id",
                                        suffixes=(f'_model', f'_human'),
                                        how="inner").drop_duplicates(), \
                               pd.merge(false_classification_neg, s_judgement, on="id",
                                        suffixes=(f'_model', f'_human'),
                                        how="inner").drop_duplicates()

    o_judgement = scores.calculate_IAA_occlusion(o_judgement, lang)
    s_judgement = scores.calculate_IAA_occlusion(s_judgement, lang)


if __name__ == '__main__':
    assert len(sys.argv) == 2
    language = sys.argv[1]
    lower_court = preprocessing.extract_prediction_test_set(OCCLUSION_PATHS["prediction"][0].format(language),
                                                            OCCLUSION_PATHS["test_sets"][0].format(language, language)
                                                            )
    lower_court_analysis(language, lower_court, f"bias_lower_court_{language}")

    baseline_df = preprocessing.extract_prediction_test_set(OCCLUSION_PATHS["prediction"][1].format(language, 1),
                                                            OCCLUSION_PATHS["test_sets"][1].format(language, language,
                                                                                                   1))
    for nr in NUMBER_OF_EXP:
        occlusion = preprocessing.extract_prediction_test_set(OCCLUSION_PATHS["prediction"][1].format(language, nr),
                                                              OCCLUSION_PATHS["test_sets"][1].format(language, language,
                                                                                                     nr)
                                                              )
        occlusion_analysis(language, occlusion, f"occlusion_{nr}_{language}", nr,
                           baseline_df[baseline_df["explainability_label"] == "Baseline"])
    qt.occlusion_analysis()
    ql.occlusion_analysis()
