import unicodedata
from typing import Any, Optional, Tuple, List, Dict

import bs4
import re

from scrc.preprocessing.section_splitter import sections
from scrc.utils.main_utils import clean_text

"""
This file is used to extract sections from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
"""


def CH_BGer(soup: Any, namespace: dict) -> Optional[Tuple[dict, List[Dict[str, str]]]]:
    """
    :param soup:        the soup parsed by bs4
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict
    """

    # As soon as one of the strings in the list (value) is encountered we switch to the corresponding section (key)
    # (?:C|c) is much faster for case insensitivity than [Cc] or (?i)c
    all_section_markers = {
        'de': {
            # "header" has no markers!
            # at some later point we can still divide rubrum into more fine-grained sections like title, judges, parties, topic
            # "title": ['Urteil vom', 'Beschluss vom', 'Entscheid vom'],
            # "judges": ['Besetzung', 'Es wirken mit', 'Bundesrichter'],
            # "parties": ['Parteien', 'Verfahrensbeteiligte', 'In Sachen'],
            # "topic": ['Gegenstand', 'betreffend'],
            "facts": [r'Sachverhalt:', r'hat sich ergeben', r'Nach Einsicht', r'A\.-'],
            "considerations": [r'Erwägung:', r'[Ii]n Erwägung', r'Erwägungen:'],
            "rulings": [r'erkennt d[\w]{2} Präsident', r'Demnach (erkennt|beschliesst)', r'beschliesst.*:\s*$', r'verfügt(\s[\wäöü]*){0,3}:\s*$', r'erk[ae]nnt(\s[\wäöü]*){0,3}:\s*$', r'Demnach verfügt[^e]'],
            "footer": [
                r'^[\-\s\w\(]*,( den| vom)?\s\d?\d\.?\s?(?:Jan(?:uar)?|Feb(?:ruar)?|Mär(?:z)?|Apr(?:il)?|Mai|Jun(?:i)?|Jul(?:i)?|Aug(?:ust)?|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?|Dez(?:ember)?)\s\d{4}([\s]*$|.*(:|Im Namen))',
                r'Im Namen des']
        },
        'fr': {
            "facts": [r'Faits\s?:', r'en fait et en droit', r'(?:V|v)u\s?:', r'A.-'],
            "considerations": [r'Considérant en (?:fait et en )?droit\s?:', r'(?:C|c)onsidérant(s?)\s?:', r'considère'],
            "rulings": [r'prononce\s?:', r'Par ces? motifs?\s?', r'ordonne\s?:'],
            "footer": [
                r'\w*,\s(le\s?)?((\d?\d)|\d\s?(er|re|e)|premier|première|deuxième|troisième)\s?(?:janv|févr|mars|avr|mai|juin|juill|août|sept|oct|nov|déc).{0,10}\d?\d?\d\d\s?(.{0,5}[A-Z]{3}|(?!.{2})|[\.])',
                r'Au nom de la Cour'
            ]
        },
        'it': {
            "facts": [r'(F|f)att(i|o)\s?:'],
            "considerations": [r'(C|c)onsiderando', r'(D|d)iritto\s?:', r'Visto:', r'Considerato'],
            "rulings": [r'(P|p)er questi motivi'],
            "footer": [
                r'\w*,\s(il\s?)?((\d?\d)|\d\s?(°))\s?(?:gen(?:naio)?|feb(?:braio)?|mar(?:zo)?|apr(?:ile)?|mag(?:gio)|giu(?:gno)?|lug(?:lio)?|ago(?:sto)?|set(?:tembre)?|ott(?:obre)?|nov(?:embre)?|dic(?:embre)?)\s?\d?\d?\d\d\s?([A-Za-z\/]{0,7}):?\s*$'
            ]
        }
    }

    if namespace['language'] not in all_section_markers:
        message = f"This function is only implemented for the languages {list(all_section_markers.keys())} so far."
        raise ValueError(message)

    section_markers = all_section_markers[namespace['language']]

    # combine multiple regex into one for each section due to performance reasons
    section_markers = dict(map(lambda kv: (kv[0], '|'.join(kv[1])), section_markers.items()))

    # normalize strings to avoid problems with umlauts
    for key, value in section_markers.items():
        section_markers[key] = unicodedata.normalize('NFC', value)
        # section_markers[key] = clean_text(value) # maybe this would solve some problems because of more cleaning

    def get_paragraphs(soup):
        """
        Get Paragraphs in the decision
        :param soup:
        :return:
        """
        divs = soup.find_all("div", class_="content")
        # we expect maximally two divs with class content
        assert len(divs) <= 2

        paragraphs = []
        heading, paragraph = None, None
        for element in divs[0]:
            if isinstance(element, bs4.element.Tag):
                text = str(element.string)
                # This is a hack to also get tags which contain other tags such as links to BGEs
                if text.strip() == 'None':
                    text = element.get_text()
                # get numerated titles such as 1. or A.
                if "." in text and len(text) < 5:
                    heading = text  # set heading for the next paragraph
                else:
                    if heading is not None:  # if we have a heading
                        paragraph = heading + " " + text  # add heading to text of the next paragraph
                    else:
                        paragraph = text
                    heading = None  # reset heading
                paragraph = clean_text(paragraph)
                if paragraph not in ['', ' ', None]:  # discard empty paragraphs
                    paragraphs.append(paragraph)
        return paragraphs

    def associate_sections(paragraphs, section_markers):
        paragraph_data = []
        section_data = {key: "" for key in sections}
        current_section = "header"
        for paragraph in paragraphs:
            # update the current section if it changed
            current_section = update_section(current_section, paragraph, section_markers)

            # construct the list of sections with associated text
            section_data[current_section] += paragraph + " "

            # construct the list of annotated paragraphs (can be used for prodigy annotation
            paragraph_data.append({"text": paragraph, "section": current_section})

        if current_section != 'footer':
            message = f"({namespace['id']}): We got stuck at section {current_section}. Please check! " \
                      f"Here you have the url to the decision: {namespace['html_url']}"
            raise ValueError(message)
        return section_data, paragraph_data

    def update_section(current_section, paragraph, section_markers):
        if current_section == 'footer':
            return current_section  # we made it to the end, hooray!
        next_section_index = sections.index(current_section) + 1
        # consider all following sections
        next_sections = sections[next_section_index:]
        for next_section in next_sections:
            marker = section_markers[next_section]
            paragraph = unicodedata.normalize('NFC', paragraph)
            if re.search(marker, paragraph):
                return next_section  # change to the next section
        return current_section  # stay at the old section

    paragraphs = get_paragraphs(soup)
    section_data, paragraph_data = associate_sections(
        paragraphs, section_markers)
    return section_data, paragraph_data

# This needs special care
# def CH_BGE(soup: Any, namespace: dict) -> Optional[dict]:
#    return CH_BGer(soup, namespace)
