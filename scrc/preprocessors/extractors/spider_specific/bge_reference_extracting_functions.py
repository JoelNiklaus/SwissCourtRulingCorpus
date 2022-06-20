import re
import json
from typing import Dict, Optional
from root import ROOT_DIR
from scrc.enums.section import Section


"""
This file is used to extract citations from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
Overview of spiders still todo: https://docs.google.com/spreadsheets/d/1FZmeUEW8in4iDxiIgixY4g0_Bbg342w-twqtiIu8eZo/edit#gid=0
"""


def XX_SPIDER(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    # This is an example spider. Just copy this method and adjust the method name and the code to add your new spider.
    pass


def CH_BGE(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract the reference to the corresponding bger file in bge
    :param sections:    the dict containing the sections per section key
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the reference as string
    """

    header = sections[Section.HEADER]

    # TODO delete code and delete json file
    """
    citation_regexes = json.loads((ROOT_DIR / 'legal_info' / 'bge_origin_bger_reference_citation.json').read_text())
    def get_combined_regexes(regex_dict):
        return re.compile("|".join([entry["regex"] for entry in regex_dict if entry["regex"]]))
    bge_reference_pattern = get_combined_regexes(citation_regexes['ruling']['bge_reference'])
    bge_reference = re.search(bge_reference_pattern, header)
        return str(bge_reference)
    """

    # create pattern for bge references: 1_456/2021 oder 1a.456/2021 oder 3a 32/2019
    pattern = re.compile('\d\D?_\d{1,3}/\d{4}|\d\D?\.\d{1,3}/\d{4}|\d\D?\s\d{1,3}/\d{4}')
    bge_reference = pattern.findall(header)[0]
    # in database decisions are saved with space, make sure pattern is the same
    bge_reference = bge_reference.replace('_', ' ')
    bge_reference = bge_reference.replace('.', ' ')
    return str(bge_reference)







