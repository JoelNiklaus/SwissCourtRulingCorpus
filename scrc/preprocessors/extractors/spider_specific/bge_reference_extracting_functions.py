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


def CH_BGE(soup: Any, namespace: dict) -> Optional[str]:
    """
    :param soup:        the soup parsed by bs4
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the string of found reference, 'no reference found' if no reference was extracted
    """

    # Add pattern "and" when two numbers are referenced
    pattern = '(\d\D?_\d{1,4}/\d{4}|\d\D?\.\d{1,4}/\d{4}|\d\D?\s\d{1,4}/\d{4}|[BIPK]\s\d{1,3}/\d{2}[^\d])'
    pattern_one = re.compile(pattern)
    pattern_two = re.compile(f"{pattern}(\s/\s|\sund\s|\set\s){pattern}")
    pattern_three = re.compile(f"{pattern}(\s/\s|\sund\s|\set\s){pattern}(\s/\s|\sund\s|\set\s){pattern}")

    # find text of first occurrence of pattern_one
    paragraph = soup.find(string=re.compile(pattern_one))
    if paragraph:
        # TODO make sure found pragraph is before Regeste
        references = re.findall(pattern_one, paragraph)
        # check for triple or double references
        bge_references_triple = re.search(pattern_three, paragraph)
        bge_references_double = re.search(pattern_two, paragraph)
        # catch wrong extracted references with counting matches and compare what is expected
        amount = 1
        if bge_references_triple:
            amount = 3
        elif bge_references_double:
            amount = 2
        if int(amount) != len(references):
            print(f"Got wrong number of references. amount : {amount}, reference: {references}")
            return 'no reference found'
        return convert_found_to_reference(references)
    else:
        return 'no reference found'
