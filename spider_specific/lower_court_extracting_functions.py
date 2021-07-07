from typing import Optional, List

import re
from scrc.utils.main_utils import clean_text

"""
This file is used to extract the lower courts from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
"""

def CH_BGer(header: str, namespace: dict) -> Optional[str]:
    """
    Extract judgement outcomes from the Federal Supreme Court of Switzerland
    :param header:     the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict
    """

    supported_languages = ['de']
    if namespace['language'] not in supported_languages:
        message = f"This function is only implemented for the languages {supported_languages} so far."
        raise ValueError(message)


    def get_lower_court_file_number(header: str, namespace: dict) -> Optional[str]:

        return None

    # make sure we don't have any nasty unicode problems
    header = clean_text(header)

    lower_court = None
    lower_court_file_number = get_lower_court_file_number(header, namespace)
    if lower_court_file_number:
        print(f'Got File number of previous court {lower_court_file_number}')
    

    return lower_court

# This needs special care
# def CH_BGE(rulings: str, namespace: dict) -> Optional[List[str]]:
#    return CH_BGer(rulings, namespace)
