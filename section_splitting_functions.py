from typing import Any, Optional

import bs4
import re

from scrc.dataset_construction.cleaner import sections
from scrc.utils.main_utils import clean_text


def CH_BGer(soup: Any, namespace: dict) -> Optional[dict]:
    """
    IMPORTANT: So far, only German is supported!
    :param soup:        the soup parsed by bs4
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict, None if not in German
    """

    if namespace['language'] != 'de':
        return None

    # As soon as one of the strings in the list (value) is encountered we switch to the corresponding section (key)
    section_markers = {
        # "header" has no markers!
        "title": ['Urteil vom'],
        "judges": ['Besetzung', 'Es wirken mit', 'Bundesrichter'],
        "parties": ['Parteien', 'Verfahrensbeteiligte', 'In Sachen'],
        "topic": ['Gegenstand', 'betreffend'],
        "situation": ['Sachverhalt', ', hat sich ergeben', 'Nach Einsicht'],
        "considerations": ['Das Bundesgericht zieht in Erwägung', 'Erwägung', 'in Erwägung', 'Erwägungen',
                           'Erwägungen'],
        "rulings": ['Demnach erkennt das Bundesgericht', 'erkennt die Präsidentin', 'erkennt der Präsident', 'erkennt'],
        "footer": [
            r'\w*,\s\d?\d\.\s(?:Jan(?:uar)?|Feb(?:ruar)?|Mär(?:z)?|Apr(?:il)?|Mai|Jun(?:i)?|Jul(?:i)?|Aug(?:ust)?|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?|Dez(?:ember)?).*']
    }

    def get_paragraphs(soup):
        """
        Get Paragraphs in the decision
        :param soup:
        :return:
        """
        divs = soup.find_all("div", class_="content")
        assert len(divs) <= 2  # we expect maximally two divs with class content

        paragraphs = []
        heading, paragraph = None, None
        for element in divs[0]:
            if isinstance(element, bs4.element.Tag):
                text = str(element.string)
                # This is a hack to also get tags which contain other tags such as links to BGEs
                if text.strip() == 'None':
                    text = element.get_text()
                if "." in text and len(text) < 5:  # get numerated titles such as 1. or A.
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
        used_sections = ["header"]  # sections which were already found
        current_section = "header"
        for paragraph in paragraphs:
            # update the current section if it changed
            current_section = update_section(current_section, paragraph, section_markers, used_sections)

            # construct the list of sections with associated text
            section_data[current_section] += paragraph + " "

            # construct the list of annotated paragraphs (can be used for prodigy annotation
            paragraph_data.append({"text": paragraph, "section": current_section})
        if len(used_sections) != len(sections):
            unused_sections = set(sections) - set(used_sections)
            raise ValueError(f"The following sections have not been used: {unused_sections}. Please check! "
                             f"Here you have the url to the decision: {namespace['html_url']}")
        return section_data, paragraph_data

    def update_section(current_section, paragraph, section_markers, used_sections):
        for section, markers in section_markers.items():
            if section not in used_sections:  # if we did not use this section yet
                for marker in markers:  # check each marker in the list
                    if re.search(marker, paragraph):
                        used_sections.append(section)  # make sure one section is only used once
                        return section  # change to the next section
        return current_section  # stay at the old section

    paragraphs = get_paragraphs(soup)
    section_data, paragraph_data = associate_sections(paragraphs, section_markers)
    return section_data
