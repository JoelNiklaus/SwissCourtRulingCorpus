from pathlib import Path
from pprint import pprint
from typing import Any, Optional
from scrc.enums.citation_type import CitationType
import json
import regex
from citation_extractor import extract_citations

from root import ROOT_DIR
from scrc.enums.language import Language

"""
This file is used to extract citations from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
Overview of spiders still todo: https://docs.google.com/spreadsheets/d/1FZmeUEW8in4iDxiIgixY4g0_Bbg342w-twqtiIu8eZo/edit#gid=0
"""


def XX_SPIDER(soup: Any, namespace: dict) -> Optional[dict]:
    valid_languages = [Language.DE, Language.IT, Language.FR]
    if namespace['language'] not in [Language.DE, Language.IT, Language.FR]:
       raise ValueError(f"This function is only implemented for the languages {valid_languages} so far.")
    citations = extract_citations(
        soup, (namespace['language'].value))
    
    return {CitationType.LAW: citations.get('laws'), CitationType.RULING: citations.get('rulings')}


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

    return {CitationType.LAW: laws, CitationType.RULING: rulings}

# This needs special care
# def CH_BGE(soup: Any, namespace: dict) -> Optional[dict]:
#    return CH_BGer(soup, namespace)
