from root import ROOT_DIR
import os
import json
import numpy as np

"""
Generates a json file with the area of each chamber based on the first digit of the law codes and court_chambers_extended.json
"""

# open the legal_info/court_chambers_extended.json file as a dictionary
# and store it in the variable 'court_chambers'
with open(os.path.join(ROOT_DIR, 'legal_info/court_chambers_extended.json'), 'r') as f:
    court_chambers = json.load(f)


areas = {
    '1': 'civil_law',
    '2': 'public_law',
    '3': 'penal_law',
    '4': 'social_law'
}


def codes_to_area(codes):
    """
    :param codes: list of str: [”2.3”, “2.4”, “3.3”]
    extracts the first digit of each element. if there are more than one unique first digit (or zero), return np.nan
    if there is one first digit, return areas[first_digit]
    """
    if len(codes) == 0:
        return np.nan
    if len(codes) == 1 and codes[0] == '':
        return np.nan
    first_digits = [code[0] for code in codes]
    if len(set(first_digits)) != 1:
        return np.nan
    # if exists, return the area corresponding to the first digit
    return areas.get(first_digits[0], np.nan)


result_dict = {}
# iterate in dictionary
for canton, canton_value in court_chambers.items():
    for court, court_value in canton_value['gerichte'].items():
        for chamber, chamber_value in court_value['kammern'].items():
            result_dict[chamber] = codes_to_area(chamber_value['Rechtsgebiete'])


print(len(result_dict))     # 422
# remove nan values
result_dict = {k: v for k, v in result_dict.items() if v is not np.nan}
print(len(result_dict))     # 218

# print result one per line
for k, v in result_dict.items():
    print(f'{k}: {v}')


# save result_dict as json with name 'chamber_to_area.json'
with open(os.path.join(ROOT_DIR, 'legal_info/chamber_to_area.json'), 'w') as f:
    json.dump(result_dict, f, indent=4)
