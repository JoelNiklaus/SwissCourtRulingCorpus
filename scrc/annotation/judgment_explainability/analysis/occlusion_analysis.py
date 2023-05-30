import sys

import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing
import scrc.annotation.judgment_explainability.analysis.utils.quantitative_analysis as qt

"""
This script prepares the results from the occlusion for the analysis.
Starts the quantitative analysis via qt.occlusion_analysis().
"""

LANGUAGES = ["de", "fr", "it"]
PERSONS = ["angela", "lynn", "thomas"]
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
OCCLUSION_PATHS = {"test_sets": ("../occlusion/lower_court_test_sets/{}/lower_court_test_set_{}.csv",
                                 "../occlusion/occlusion_test_sets/{}/occlusion_test_set_{}_exp_{}.csv"),
                   "prediction": ("../occlusion/lower_court_predictions/{}/predictions_test.csv",
                                  "../occlusion/occlusion_predictions/{}_{}/predictions_test.csv")}

NUMBER_OF_EXP = [1, 2, 3, 4]

if __name__ == '__main__':
    """
    Argument handling, dataset extraction.
    """
    assert len(sys.argv) == 2
    language = sys.argv[1]
    baseline_df = preprocessing.extract_prediction_test_set(OCCLUSION_PATHS["prediction"][1].format(language, 1),
                                                            OCCLUSION_PATHS["test_sets"][1].format(language, language,
                                                                                                   1))
    for nr in NUMBER_OF_EXP:
        occlusion = preprocessing.extract_prediction_test_set(OCCLUSION_PATHS["prediction"][1].format(language, nr),
                                                              OCCLUSION_PATHS["test_sets"][1].format(language, language,
                                                                                                     nr))
        if nr in NUMBER_OF_EXP[1:]:
            occlusion = occlusion.append(baseline_df)
        preprocessing.occlusion_preprocessing(language, occlusion, f"occlusion_{nr}_{language}")

    qt.occlusion_analysis()
