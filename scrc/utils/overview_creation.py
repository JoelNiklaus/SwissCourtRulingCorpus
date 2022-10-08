from root import ROOT_DIR
import os
import sys
import csv
from tqdm import tqdm
from scrc.utils.court_names import get_all_courts
from scrc.enums.split import Split


csv.field_size_limit(sys.maxsize)

## TODO: move to dataset creation

#   creates an overview of the size of the generated datasets and exports them in a csv file
def create_overview(include_all=True, path=str(self.datasets_subdir), export_path=str(self.data_dir), export_name="overview.csv"):
    courts_av = os.listdir(os.path.join(ROOT_DIR, path))

    #   stores the overview in a list of dicts
    courts_data = []

    #   store the number of rows of each file in a dict for each court
    for court in tqdm(courts_av):
        court_data = {"name": court}
        for key in [split.value for split in Split]:    # ["all", "val", "test", "train", "secret_test"]
            try:
                with open(os.path.join(ROOT_DIR, path, court, f"{key}.csv"), "r") as f:
                    reader = csv.reader(f)
                    court_data[key] = len(list(reader)) - 1  # -1 because of header
            except FileNotFoundError:
                court_data[key] = -2
        court_data['created'] = True
        courts_data.append(court_data)

    #   add courts that are not in the folder
    if include_all:
        for court in get_all_courts():
            if court not in courts_av:
                court_data = {"name": court, 'created': False}
                courts_data.append(court_data)

    #   export to csv
    with open(os.path.join(ROOT_DIR, export_path, export_name), "w") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "all", "val", "test", "train", "secret_test", "created"])
        writer.writeheader()
        writer.writerows(courts_data)
    print("Overview created and exported to ", os.path.join(ROOT_DIR, export_path, export_name))


if __name__ == '__main__':
    create_overview()
