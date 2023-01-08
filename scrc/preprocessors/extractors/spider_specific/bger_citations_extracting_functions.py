import bs4
import re
from typing import List, Dict, Union
from typing import Any, Optional
from scrc.enums.section import Section


"""
This file is used to extract bger citations.
"""


def convert_found_to_reference(references):
    cit_list = []
    for item in references:
        pattern = re.compile('[BIPK]\s\d{1,3}/\d{2}[^\d]')
        if re.match(pattern, item):
            item = item[:-1]
        citation = item.strip()
        citation = citation.replace(' ', '_')
        citation = citation.replace('.', '_')
        # replace references with pattern like 1, 124/1996
        citation = citation.replace(',', '')
        cit_list.append(citation)
    return "-".join(cit_list)


def XX_SPIDER(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:
    """
    # This is an example spider. Just copy this method and adjust the method name and the code to add your new spider.
    pass


def CH_BGE(full_text: str, namespace: dict) -> Optional[str]:

    # Add pattern "unt" or "et" or "/" when multiple numbers are referenced
    pattern = '(\d\D?_\d{1,4}/\d{4}|\d\D?\.\d{1,4}/\d{4}|\d\D?\s\d{1,4}/\d{4}|[BIPK]\s\d{1,3}/\d{2}[^\d])'
    pattern_one = re.compile(pattern)

    # find all occurrence of pattern_one
    found_refs = re.findall(re.compile(pattern_one), full_text)
    if found_refs:
        return convert_found_to_reference(found_refs)
    else:
        return 'no citations found'
