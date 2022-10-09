from root import ROOT_DIR
import os
import sys
import csv
from tqdm import tqdm
from scrc.utils.court_names import get_all_courts
from scrc.enums.split import Split
from scrc.utils.main_utils import get_config

csv.field_size_limit(sys.maxsize)
config = get_config()
data_dir = f"{str(ROOT_DIR)}/{config['dir']['data_dir']}"
datasets_subdir = f"{data_dir}/{config['dir']['datasets_subdir']}"


#   creates an overview of the size of the generated datasets and exports them in a csv file
def create_overview(path, export_path, export_name="overview", include_all=False):
    """
    :param path:            path to the court dataset folders
    :param include_all:     if True, all courts are included in the overview otherwise only the created courts
    """
    courts_av_tmp = os.listdir(path)
    courts_av = []
    # filtering to only have dir's
    for s in courts_av_tmp:
        if not os.path.isfile(f"{path}/{s}"):
            courts_av.append(s)

    #   stores the overview in a list of dicts
    courts_data = []

    # store the number of rows of each file in a dict for each court
    for court in tqdm(courts_av):
        court_data = {"name": court}
        for key in [split.value for split in Split]:  # ["all", "val", "test", "train", "secret_test"]
            try:
                with open(os.path.join(path, court, f"{key}.csv"), "r") as f:
                    reader = csv.reader(f)
                    court_data[key] = len(list(reader)) - 1  # -1 because of header
            except FileNotFoundError:
                court_data[key] = -2
        court_data['created'] = True
        courts_data.append(court_data)

    # add courts that are not in the folder
    if include_all:
        for court in get_all_courts():
            if court not in courts_av:
                court_data = {"name": court, 'created': False}
                courts_data.append(court_data)

    # check if export file already exists and increment the name index if it does
    counter = 1
    while os.path.exists(f"{export_path}/{export_name}_{counter}.csv"):
        counter += 1
    export_name = f"{export_name}_{counter}.csv"

    # export to csv
    with open(os.path.join(export_path, export_name), "w") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "all", "val", "test", "train", "secret_test", "created"])
        writer.writeheader()
        writer.writerows(courts_data)
    print("Overview created and exported to ", os.path.join(export_path, export_name))

