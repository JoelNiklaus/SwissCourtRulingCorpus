from typing import List, Dict, Union
import bs4
import re

from typing import Any, Optional
from scrc.enums.section import Section


"""
This file is used to extract citations from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
Overview of spiders still todo: https://docs.google.com/spreadsheets/d/1FZmeUEW8in4iDxiIgixY4g0_Bbg342w-twqtiIu8eZo/edit#gid=0
"""

def XX_SPIDER(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    # This is an example spider. Just copy this method and adjust the method name and the code to add your new spider.
    pass


def CH_BGE(soup: Any, namespace: dict) -> Optional[str]:
    """
    :param soup:        the soup parsed by bs4
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict, None if not in German
    """

    pattern = re.compile('\d\D?_\d{1,3}/\d{4}|\d\D?\.\d{1,3}/\d{4}|\d\D?\s\d{1,3}/\d{4}')
    bge_references = soup.find(string=re.compile(pattern))
    if bge_references:
        # only consider first entry
        bge_reference = re.search(pattern, bge_references).group()
        bge_reference = bge_reference.replace('_', ' ')
        bge_reference = bge_reference.replace('.', ' ')
        return bge_reference
    else:
        bge_reference = 'no reference found'
    return bge_reference










