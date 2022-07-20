from pathlib import Path
from pprint import pprint
from typing import Any, Optional, Tuple

import pandas as pd
from scrc.data_classes.law_citation import LawCitation
from scrc.data_classes.ruling_citation import RulingCitation
from scrc.enums.citation_type import CitationType
from citation_extractor import extract_citations

from root import ROOT_DIR
from scrc.enums.language import Language

"""
This file is used to extract citations from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
Overview of spiders still todo: https://docs.google.com/spreadsheets/d/1FZmeUEW8in4iDxiIgixY4g0_Bbg342w-twqtiIu8eZo/edit#gid=0
"""

available_laws = pd.read_json((ROOT_DIR / "corpora") / "lexfind.jsonl", lines=True) # Doesnt include BGG, ATSG, LTF

def check_if_convertible(laws, rulings, language: Language) -> Tuple[list, list]:
    """ Test if the citations can be converted into the dataclasses. If not, then it is probable, that the citations are not correctly extracted (e.g. missing the law) and can be ignored """
    valid_laws = set()
    valid_rulings = set()
    language_str = language.value
    for law in laws:
        try:
            _ = LawCitation(law['text'], language_str, available_laws )
            valid_laws.add(law)
        except BaseException as e:
            print(e)
            continue
        
    for ruling in rulings:
        try:
            _ = RulingCitation(ruling['text'], language_str)
            valid_rulings.add(ruling)
        except:
            continue
        
    return (list(valid_laws), list(valid_rulings))

def XX_SPIDER(soup: Any, namespace: dict) -> Optional[dict]:
    valid_languages = [Language.DE, Language.IT, Language.FR]
    if namespace['language'] not in [Language.DE, Language.IT, Language.FR]:
       raise ValueError(f"This function is only implemented for the languages {valid_languages} so far.")
    citations = extract_citations(
        soup, (namespace['language'].value))
    laws, rulings = check_if_convertible(citations.get('laws'), citations.get('rulings'), namespace['language'])
    return {CitationType.LAW: laws, CitationType.RULING: rulings}


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
            rulings.append(
                {"type": "bge", "url": bge['href'], "text": bge.string})

    laws, rulings = check_if_convertible(laws, rulings, namespace['language'])
    return {CitationType.LAW: laws, CitationType.RULING: rulings}

# This needs special care
# def CH_BGE(soup: Any, namespace: dict) -> Optional[dict]:
#    return CH_BGer(soup, namespace)
