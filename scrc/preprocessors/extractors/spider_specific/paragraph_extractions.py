
from typing import Optional, List, Dict, Union

import bs4
import re
import unicodedata

from bs4 import BeautifulSoup

from scrc.enums.section import Section
from scrc.utils.main_utils import clean_text
from scrc.utils.main_utils import get_paragraphs_unified


def XX_SPIDER(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    if isinstance(decision, str):
        return get_pdf_paragraphs(decision)
    return get_paragraphs_unified(decision)


def get_pdf_paragraphs(soup: str) -> list:
    """
    Get the paragraphs of a decision
    :param soup:    the string extracted of the pdf
    :return:        a list of paragraphs
    """

    paragraphs = []
    # remove spaces between two line breaks
    soup = re.sub('\\n +\\n', '\\n\\n', soup)
    # split the lines when there are two line breaks
    lines = soup.split('\n\n')
    for element in lines:
        element = element.replace('  ', ' ')
        paragraph = clean_text(element)
        if paragraph not in ['', ' ', None]:  # discard empty paragraphs
            
            paragraphs.append(paragraph)
    return paragraphs






