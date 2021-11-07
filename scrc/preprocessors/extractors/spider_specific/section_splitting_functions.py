import unicodedata
import collections
from typing import Optional, List, Dict, Union

import bs4
import re
import json

from scrc.enums.language import Language
from scrc.enums.section import Section
from scrc.utils.main_utils import clean_text

"""
This file is used to extract sections from decisions sorted by spiders.
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


def VD_Omni(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    # print("\nVD_Omni\n")

    def get_paragraphs(soup):
        """
        Get composition of the decision
        :param soup:
        :return:
        """
        # by some bruteforce I have found out that the html files in VD_Omni all have a single div with one of the
        # following classes:
        possible_main_div_classes = ["WordSection1", "Section1"]
        # Iterate over this list and find out the current decision is using which class type
        for main_div_class in possible_main_div_classes:

            div = soup.find_all("div", class_=main_div_class)
            if (len(div) != 0):
                class_found = main_div_class
                break
        # We expect to have only one main div
        assert len(div) == 1
        # If the divs is empty raise an error
        if (len(div) == 0):
            message = f"The main div has an unseen class type"
            raise ValueError(message)
        paragraphs = []
        # paragraph = None
        for element in div:
            if isinstance(element, bs4.element.Tag):
                text = str(element.string)
                # This is a hack to also get tags which contain other tags such as links to BGEs
                if text.strip() == 'None':
                    text = element.get_text()
                paragraph = clean_text(text)
                if paragraph not in ['', ' ', None]:  # discard empty paragraphs
                    paragraphs.append(paragraph)
        return paragraphs
    def get_composition_candidates(paragraphs, RegEx):

        # did we miss composition of some of the decisions?
        missed_cnt = 0
        composition_candidate = None
        for paragraph in paragraphs:
            composition = RegEx.search(paragraph)
            # If we did not find the RegEx in the paragraph
            if composition is None:
                missed_cnt += 1
                continue
            # If we found it, we will take a 2*range substring and use it as composition candidate
            starting_index = composition.start()
            # Hopefully the composition length will not be more than 400 characters
            range = 200
            # starting index
            s_indx = starting_index - range if (starting_index > range) else 0
            # ending index
            e_indx = starting_index + range if (len(paragraph) > starting_index + range) else len(paragraph)
            # extract the compostion
            composition_candidate = paragraph[s_indx:e_indx]
            # Store all the compositions in a signle list

        return composition_candidate, missed_cnt

    paragraphs_by_section = {section: [] for section in Section}
    paragraphs = get_paragraphs(decision)
    # Currently, we search the whole decision for the following keywords: president, presidence, compose, composition
    composition_RegEx = re.compile(r'[P,p]r[é,e]siden[t,c]|compos[é,e] | [C,c]omposition')
    composition_candidate, missed_cnt = get_composition_candidates(paragraphs, composition_RegEx)

    if missed_cnt != 0 or composition_candidate is None:
        # We write all the missed descisions to a file to take a closer look and improve the efficiency
        f = open("VD_Omni_missed_decisions.txt", "a")
        f.write('%s' % paragraphs)
        f.write('\n-------------------------------------------------------------\n')
        f.close()
        message = f"We have missed the judicial people for some decisions"
        raise ValueError(message)

    paragraphs_by_section[Section.HEADER] = composition_candidate
    paragraphs_by_section[Section.FACTS] = paragraphs
    return paragraphs_by_section




def CH_BGer(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """


    # As soon as one of the strings in the list (regexes) is encountered we switch to the corresponding section (key)
    # (?:C|c) is much faster for case insensitivity than [Cc] or (?i)c
    all_section_markers = {
        Language.DE: {
            # "header" has no markers!
            # at some later point we can still divide rubrum into more fine-grained sections like title, judges, parties, topic
            # "title": ['Urteil vom', 'Beschluss vom', 'Entscheid vom'],
            # "judges": ['Besetzung', 'Es wirken mit', 'Bundesrichter'],
            # "parties": ['Parteien', 'Verfahrensbeteiligte', 'In Sachen'],
            # "topic": ['Gegenstand', 'betreffend'],
            Section.FACTS: [r'Sachverhalt:', r'hat sich ergeben', r'Nach Einsicht', r'A\.-'],
            Section.CONSIDERATIONS: [r'Erwägung:', r'[Ii]n Erwägung', r'Erwägungen:'],
            Section.RULINGS: [r'erkennt d[\w]{2} Präsident', r'Demnach (erkennt|beschliesst)', r'beschliesst.*:\s*$',
                              r'verfügt(\s[\wäöü]*){0,3}:\s*$', r'erk[ae]nnt(\s[\wäöü]*){0,3}:\s*$',
                              r'Demnach verfügt[^e]'],
            Section.FOOTER: [
                r'^[\-\s\w\(]*,( den| vom)?\s\d?\d\.?\s?(?:Jan(?:uar)?|Feb(?:ruar)?|Mär(?:z)?|Apr(?:il)?|Mai|Jun(?:i)?|Jul(?:i)?|Aug(?:ust)?|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?|Dez(?:ember)?)\s\d{4}([\s]*$|.*(:|Im Namen))',
                r'Im Namen des']
        },
        Language.FR: {
            Section.FACTS: [r'Faits\s?:', r'en fait et en droit', r'(?:V|v)u\s?:', r'A.-'],
            Section.CONSIDERATIONS: [r'Considérant en (?:fait et en )?droit\s?:', r'(?:C|c)onsidérant(s?)\s?:',
                                     r'considère'],
            Section.RULINGS: [r'prononce\s?:', r'Par ces? motifs?\s?', r'ordonne\s?:'],
            Section.FOOTER: [
                r'\w*,\s(le\s?)?((\d?\d)|\d\s?(er|re|e)|premier|première|deuxième|troisième)\s?(?:janv|févr|mars|avr|mai|juin|juill|août|sept|oct|nov|déc).{0,10}\d?\d?\d\d\s?(.{0,5}[A-Z]{3}|(?!.{2})|[\.])',
                r'Au nom de la Cour'
            ]
        },
        Language.IT: {
            Section.FACTS: [r'(F|f)att(i|o)\s?:'],
            Section.CONSIDERATIONS: [r'(C|c)onsiderando', r'(D|d)iritto\s?:', r'Visto:', r'Considerato'],
            Section.RULINGS: [r'(P|p)er questi motivi'],
            Section.FOOTER: [
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
    for section, regexes in section_markers.items():
        section_markers[section] = unicodedata.normalize('NFC', regexes)
        # section_markers[key] = clean_text(regexes) # maybe this would solve some problems because of more cleaning

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

    paragraphs = get_paragraphs(decision)

    return associate_sections(paragraphs, section_markers, namespace)


def associate_sections(paragraphs: List[str], section_markers, namespace: dict):
    paragraphs_by_section = {section: [] for section in Section}
    current_section = Section.HEADER
    for paragraph in paragraphs:
        # update the current section if it changed
        current_section = update_section(current_section, paragraph, section_markers)

        # add paragraph to the list of paragraphs
        paragraphs_by_section[current_section].append(paragraph)
    if current_section != Section.FOOTER:
        message = f"({namespace['id']}): We got stuck at section {current_section}. Please check! " \
                  f"Here you have the url to the decision: {namespace['html_url']}"
        raise ValueError(message)
    return paragraphs_by_section


def update_section(current_section: Section, paragraph: str, section_markers) -> Section:
    if current_section == Section.FOOTER:
        return current_section  # we made it to the end, hooray!
    sections = list(Section)
    next_section_index = sections.index(current_section) + 1
    # consider all following sections
    next_sections = sections[next_section_index:]
    for next_section in next_sections:
        marker = section_markers[next_section]
        paragraph = unicodedata.normalize('NFC', paragraph)  # if we don't do this, we get weird matching behaviour
        if re.search(marker, paragraph):
            return next_section  # change to the next section
    return current_section  # stay at the old section

# This needs special care
# def CH_BGE(decision: Any, namespace: dict) -> Optional[dict]:
#    return CH_BGer(decision, namespace)
