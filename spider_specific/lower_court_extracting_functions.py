from typing import Match, Optional, List
from pathlib import Path
import re
import unicodedata
import json
import pandas as pd
from sqlalchemy.engine.base import Engine
from scrc.utils.main_utils import clean_text

"""
This file is used to extract the lower courts from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
"""



def CH_BGer(header: str, namespace: dict, engine: Engine) -> Optional[str]:
    """
    Extract lower courts from decisions of the Federal Supreme Court of Switzerland
    :param header:     the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict
    """
    supported_languages = ['de']
    if namespace['language'] not in supported_languages:
        message = f"This function is only implemented for the languages {supported_languages} so far."
        raise ValueError(message)


    file_number_regex = [r"""(?<=\() # find opening paranthesis without including it in result
        (?P<ID>[A-Z1-9]{0,4})
        (\.|\s)? # Split by point or whitespace or nothing
        (?P<YEAR>\d{0,4})
        (\.|\s)? # Split by point or whitespace or nothing
        (?P<NUMBER>\d{0,8})
        (?=\)) # find closing paranthesis without including it in result
    """, 
    r"""
    (?<=\() # find opening paranthesis without including it in result
    [A-Z]{2}
    \d{6}
    (?=\)) # find closing paranthesis without including it in result
    """]
     # combine multiple regex into one for each section due to performance reasons
    file_number_regex ='?|'.join(file_number_regex)
    # normalize strings to avoid problems with umlauts
    file_number_regex = unicodedata.normalize('NFC', file_number_regex)
        
    def get_lower_court_file_number(header: str, namespace: dict) -> Optional[Match]:
        result = re.search(file_number_regex, header, re.X)
        return result

    def get_lower_court_by_file_number(file_number: Match) -> Optional[str]:
        languages = ['de', 'fr', 'it']
        chamber = None
        
        for lang in languages:
            if (file_number.group('YEAR')):
                id = file_number.group('ID')
                year = file_number.group('YEAR')
                number = file_number.group('Number')
                chamber = pd.read_sql(f"SELECT chamber FROM {lang} WHERE file_number LIKE '{id}%{year}%{number}'", engine.connect())
            else: 
                chamber = pd.read_sql(f"SELECT chamber FROM {lang} WHERE file_number = '{file_number.group()}'", engine.connect())
            print(chamber)
        
        return None

    # make sure we don't have any nasty unicode problems
    header = clean_text(header)

    lower_court = None
    lower_court_file_number = get_lower_court_file_number(header, namespace)
    if lower_court_file_number:
        print(f'Got File number of previous court {lower_court_file_number} \n{header}\n{namespace["html_url"]}')
        lower_court = get_lower_court_by_file_number(lower_court_file_number)
        input()
    else:
        print(f'Got no File number of previuous court for \n{header}\n{namespace["html_url"]}')
    
    return lower_court

# This needs special care
# def CH_BGE(rulings: str, namespace: dict) -> Optional[List[str]]:
#    return CH_BGer(rulings, namespace)
