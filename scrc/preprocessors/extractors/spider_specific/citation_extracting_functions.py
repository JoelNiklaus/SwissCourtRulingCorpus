from pathlib import Path
from pprint import pprint
from typing import Any, Optional

import json

import regex

from root import ROOT_DIR

"""
This file is used to extract citations from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
Overview of spiders still todo: https://docs.google.com/spreadsheets/d/1FZmeUEW8in4iDxiIgixY4g0_Bbg342w-twqtiIu8eZo/edit#gid=0
"""


def XX_SPIDER(soup: Any, namespace: dict) -> Optional[dict]:
    # This is an example spider. Just copy this method and adjust the method name and the code to add your new spider.
    pass


def SH_OG(soup: Any, namespace: dict) -> Optional[dict]:
    print(soup)

    soup += """
    dass der vorliegende Begründungsmangel offensichtlich ist, weshalb auf die Beschwerde in Anwendung von Art. 108 Abs. 1 lit. b BGG nicht einzutreten ist, 
dass von der Erhebung von Gerichtskosten für das bundesgerichtliche Verfahren umständehalber abzusehen ist (Art. 66 Abs. 1 Satz 2 BGG), 
dass in den Fällen des Art. 108 Abs. 1 BGG das vereinfachte Verfahren zum Zuge kommt und die Abteilungspräsidentin zuständig ist,  
"""

    def get_combined_regexes(regex_dict, language):
        return regex.compile("|".join([entry["regex"] for entry in regex_dict[language] if entry["regex"]]))

    language = 'de'
    citation_regexes = json.loads((ROOT_DIR / 'citation_regexes.json').read_text())
    pprint(citation_regexes)
    print("BGE")
    for match in regex.findall(get_combined_regexes(citation_regexes['ruling']['BGE'], language), soup):
        print(match)
    print("Bger")
    for match in regex.findall(get_combined_regexes(citation_regexes['ruling']['Bger'], language), soup):
        print(match)
    print("law")
    for match in regex.findall(get_combined_regexes(citation_regexes['law'], language), soup):
        print(match)
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

    return {"laws": laws, "rulings": rulings}

# This needs special care
# def CH_BGE(soup: Any, namespace: dict) -> Optional[dict]:
#    return CH_BGer(soup, namespace)
