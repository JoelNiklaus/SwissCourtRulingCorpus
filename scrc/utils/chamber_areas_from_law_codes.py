import os
import json
import numpy as np

from root import ROOT_DIR

# Mapping of first digits to areas
areas = {
    '1': 'civil_law',
    '2': 'public_law',
    '3': 'penal_law',
    '4': 'social_law'
}

with open(os.path.join(ROOT_DIR, 'legal_info/law_areas.json'), 'r') as f:
    law_areas = json.load(f)

with open(os.path.join(ROOT_DIR, 'legal_info/court_chambers_extended.json'), 'r') as f:
    court_chambers = json.load(f)

def sub_code_to_area(code):
    """
    input: code of the form '1.4.'
    Returns the name of the sub-area corresponding to the given code.
    """
    code = code.split('.')
    name = law_areas['law_areas'][code[0]]['law_areas'][code[1]]['de']
    return name

def codes_to_area(codes):
    """
    Returns the area corresponding to the first digit of each element in the given list of codes.
    If there are more than one unique first digit (or zero), returns np.nan.
    If the list is empty or contains only an empty string, returns np.nan.
    """
    if len(codes) == 0 or (len(codes) == 1 and codes[0] == ''):
        return np.nan
    first_digits = [code[0] for code in codes]
    if len(set(first_digits)) != 1:
        return np.nan
    # if exists, return the area corresponding to the first digit
    return areas.get(first_digits[0], np.nan)

def map_chamber_codes(chamber, chamber_value):
    result_dict = {}
    sub_result_dict = {}
    codes = chamber_value['Rechtsgebiete']
    result_dict[chamber] = codes_to_area(codes)
    if len(codes) == 1 and codes[0] != '':
        sub_result_dict[chamber] = sub_code_to_area(codes[0])
    return result_dict, sub_result_dict

def filter_nan_values(dictionary):
    return {k: v for k, v in dictionary.items() if v is not np.nan}

def save_json(filename, data):
    with open(os.path.join(ROOT_DIR, filename), 'w') as f:
        json.dump(data, f, indent=4)
    print("Saved file to '{}'".format(filename))

def create(court_chambers):
    result_dict = {}
    sub_result_dict = {}
    for canton, canton_value in court_chambers.items():
        for court, court_value in canton_value['gerichte'].items():
            for chamber, chamber_value in court_value['kammern'].items():
                result, sub_result = map_chamber_codes(chamber, chamber_value)
                result_dict.update(result)
                sub_result_dict.update(sub_result)
    result_dict = filter_nan_values(result_dict)
    sub_result_dict = filter_nan_values(sub_result_dict)
    save_json('legal_info/chamber_to_area_v.json', result_dict)
    save_json('legal_info/chamber_to_sub_area_v.json', sub_result_dict)

if __name__ == '__main__':
    create(court_chambers)