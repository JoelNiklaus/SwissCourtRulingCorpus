import bs4
import re
from typing import List, Dict, Union
from typing import Any, Optional
from scrc.enums.section import Section


"""
This file is used to extract references to a bger.
"""


def convert_found_to_reference(references):
    ref_list = []
    for item in references:
        pattern = re.compile('[BIPK]\s\d{1,3}/\d{2}[^\d]')
        if re.match(pattern, item):
            item = item[:-1]
        bge_reference = item.strip()
        bge_reference = bge_reference.replace(' ', '_')
        bge_reference = bge_reference.replace('.', '_')
        # replace references with pattern like 1, 124/1996
        bge_reference = bge_reference.replace(',', '')
        ref_list.append(bge_reference)
    return "-".join(ref_list)


def XX_SPIDER(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:
    """
    # This is an example spider. Just copy this method and adjust the method name and the code to add your new spider.
    pass


# def CH_BGE(soup: Any, namespace: dict) -> Optional[str]:
    """
    :param soup:        the soup parsed by bs4
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the string of found reference, 'no reference found' if no reference was extracted
    """

def CH_BGE(header: str, namespace: dict) -> Optional[str]:

    # Add pattern "unt" or "et" or "/" when multiple numbers are referenced
    pattern = '(\d\D?_\d{1,4}/\d{4}|\d\D?\.\d{1,4}/\d{4}|\d\D?\s\d{1,4}/\d{4}|[BIPK]\s\d{1,3}/\d{2}[^\d])'
    pattern_one = re.compile(pattern)
    pattern_two = re.compile(f"{pattern}(\s/\s|\sund\s|\set\s){pattern}")
    pattern_three = re.compile(f"{pattern}(\s/\s|\sund\s|\set\s){pattern}(\s/\s|\sund\s|\set\s){pattern}")

    # find text of first occurrence of pattern_one
    # paragraph = soup.find(string=re.compile(pattern_one))
    found_refs = re.findall(re.compile(pattern_one), header)
    if found_refs:
        amount = len(found_refs)
        bge_references_triple = re.search(pattern_three, header)
        bge_references_double = re.search(pattern_two, header)
        # catch wrong extracted references with counting matches and compare what is expected
        if bge_references_triple:
            amount = 3
        elif bge_references_double:
            amount = 2
        if amount != len(found_refs):
            print(f"Got wrong number of references. amount : {amount}, reference: {found_refs}")
            return 'no reference found'
        return convert_found_to_reference(found_refs)
    else:
        return 'no reference found'
