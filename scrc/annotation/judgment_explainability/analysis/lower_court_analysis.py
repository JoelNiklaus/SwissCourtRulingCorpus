import sys

import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing
import scrc.annotation.judgment_explainability.analysis.utils.quantitative_analysis as qt

LANGUAGES = ["de", "fr", "it"]
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
    # Lower court analysis
    lower_court_df = preprocessing.extract_prediction_test_set(OCCLUSION_PATHS["prediction"][0].format(language),
                                                               OCCLUSION_PATHS["test_sets"][0].format(language,
                                                                                                      language)
                                                               )
    preprocessing.occlusion_preprocessing(language, lower_court_df, f"bias_lower_court_{language}")
    qt.lower_court_analysis()
