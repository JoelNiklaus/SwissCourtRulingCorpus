"""
@InProceedings{Kueppers_2020_CVPR_Workshops,
   author = {Küppers, Fabian and Kronenberger, Jan and Shantia, Amirhossein and Haselhoff, Anselm},
   title = {Multivariate Confidence Calibration for Object Detection},
   booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
   month = {June},
   year = {2020}
}
https://github.com/fabiankueppers/calibration-framework#calibration-framework
"""
import numpy as np
import pandas as pd
from netcal.scaling import TemperatureScaling
from scrc.annotation.judgment_explainability.annotations.preprocessing_functions import read_csv

df = read_csv("../../prodigy_dataset_creation/sjp/finetune/xlm-roberta-base-hierarchical/de,fr,it,en/3/de/predictions.csv", "index")
df["prediction"] = np.where(df["prediction"] == "dismissal", 0, 1)
ground_truth=np.array(df["prediction"].values)
confidences=np.array(df[["prediction","confidence"]].values)


temperature = TemperatureScaling()
temperature.fit(confidences, ground_truth)
calibrated = temperature.transform(confidences)
print(df[["prediction","confidence"]])
print(pd.DataFrame(calibrated))