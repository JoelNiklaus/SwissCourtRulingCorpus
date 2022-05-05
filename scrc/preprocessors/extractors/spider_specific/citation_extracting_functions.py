from pathlib import Path
from pprint import pprint
from typing import Any, Optional
from scrc.enums.citation_type import CitationType
import json
import regex

from root import ROOT_DIR

"""
This file is used to extract citations from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
Overview of spiders still todo: https://docs.google.com/spreadsheets/d/1FZmeUEW8in4iDxiIgixY4g0_Bbg342w-twqtiIu8eZo/edit#gid=0
"""


def XX_SPIDER(soup: Any, namespace: dict) -> Optional[dict]:
    def get_combined_regexes(regex_dict, language):
        return regex.compile("|".join([entry["regex"] for entry in regex_dict[language] if entry["regex"]]))
    
    
    citation_regexes = json.loads((ROOT_DIR / 'legal_info' / 'citation_regexes.json').read_text())
    pprint(citation_regexes)
    rulings = []
    laws = []
    print("BGE")
    for match in regex.findall(get_combined_regexes(citation_regexes['ruling']['BGE'], namespace['language']), soup):
        rulings.append(match.group(0))
        print(match)
    print("Bger")
    for match in regex.findall(get_combined_regexes(citation_regexes['ruling']['Bger'], namespace['language']), soup):
        rulings.append(match.group(0))
        print(match)
    print("law")
    for match in regex.findall(get_combined_regexes(citation_regexes['law'], namespace['language']), soup):
        laws.append(match.group(0))
        print(match)
    print({CitationType.LAW: laws, CitationType.RULING: rulings})
    exit()




# TODO regexes überprüfen mit Zitierungen des Bundesgerichts

def CH_BGer(soup: Any, namespace: dict) -> Optional[dict]:
    """
    :param soup:        the soup parsed by bs4
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict, None if not in German
    """
    law_key = 'artref'
    bge_key = 'bgeref_id'

    laws, rulings = [], []

    for law in soup.find_all("span", class_=law_key):
        if law.string:  # make sure it is not empty or None
            laws.append({"text": law.string})

    for bge in soup.find_all("a", class_=bge_key):
        if bge.string:  # make sure it is not empty or None
            rulings.append({"type": "bge", "url": bge['href'], "text": bge.string})

    print({CitationType.LAW: laws, CitationType.RULING: rulings})
    exit()
    return {CitationType.LAW: laws, CitationType.RULING: rulings}

# This needs special care
# def CH_BGE(soup: Any, namespace: dict) -> Optional[dict]:
#    return CH_BGer(soup, namespace)
