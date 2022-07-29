import bs4
import re
from typing import List, Dict, Union
from typing import Any, Optional
from scrc.enums.section import Section


"""
This file is used to extract references to a bger.
"""

def XX_SPIDER(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:
    """
    # This is an example spider. Just copy this method and adjust the method name and the code to add your new spider.
    pass


def CH_BGE(soup: Any, namespace: dict) -> Optional[str]:
    """
    :param soup:        the soup parsed by bs4
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the string of found reference, 'no reference found' if no reference was extracted
    """

    pattern = re.compile('\d\D?_\d{1,4}/\d{4}|\d\D?\.\d{1,4}/\d{4}|\d\D?\s\d{1,4}/\d{4}')
    bge_references = soup.find(string=re.compile(pattern))
    if bge_references:
        # only consider first entry
        bge_reference = re.search(pattern, bge_references).group()
        bge_reference = bge_reference.replace('_', ' ')
        bge_reference = bge_reference.replace('.', ' ')
        # replace references with pattern like 1, 124/1996
        bge_reference = bge_reference.replace(',', '')
        return bge_reference
    else:
        return 'no reference found'