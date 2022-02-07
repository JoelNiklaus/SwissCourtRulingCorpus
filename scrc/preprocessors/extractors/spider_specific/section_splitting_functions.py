import unicodedata
from typing import Optional, List, Dict, Union

import bs4
import re

from scrc.enums.language import Language
from scrc.enums.section import Section
from scrc.utils.main_utils import clean_text
from scrc.utils.log_utils import get_logger

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

def JU_Gerichte(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    all_section_markers = {
        Language.FR: {
	    Section.HEADER: [r"COUR CONSTITUTIONNELLE|TRIBUNAL DE PREMIÈRE INSTANCE|CONSIDERANTS DU JUGEMENT|CHAMBRE PÉNALE DES RECOURS|COUR CIVILE|COUR ADMINISTRATIVE|COUR DES ASSURANCES|COUR DES POURSUITES ET FAILLITES|COUR PÉNALE"],
            Section.FACTS: [r'En fait:|______|EN PROCEDURE ET EN FAIT'],
            Section.CONSIDERATIONS: [r'En droit:|EN DROIT'],
            Section.RULINGS: [r'PAR CES MOTIFS LA COUR DES ASSURANCES|PAR CES MOTIFS LA COUR ADMINISTRATIVE|PAR CES MOTIFS LA CHAMBRE PÉNALE DES RECOURS|PAR CES MOTIFS LA COUR CIVILE|LE JUGE PENAL DU TRIBUNAL DE PREMIERE INSTANCE Après exposé oral des motifs|PAR CES MOTIFS LE PRÉSIDENT DE LA COUR CONSTITUTIONNELLE|PAR CES MOTIFS LA COUR CONSTITUTIONNELLE|PAR CES MOTIFS LA COUR PÉNALE|PAR CES MOTIFS LA COUR DES POURSUITES ET FAILLITES'],
            Section.FOOTER: [r"Communication concernant les moyens de recours|A notifier:"]
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

    paragraphs = get_pdf_paragraphs(decision)

    return associate_sections(paragraphs, section_markers, namespace)

def UR_Gerichte(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    all_section_markers = {
        Language.DE: {
            Section.FACTS: [r'Sachverhalt:'],
            Section.CONSIDERATIONS: [r'Aus den Erwägungen:', r'Aus den Erwägungen des Bundesgerichts:', r'Erwägungen:']
        }
    }

    sections_found = {}
    for lang in all_section_markers:
        for sect in all_section_markers[lang]:
            for reg in (all_section_markers[lang])[sect]:
                matches = re.finditer(reg, decision, re.MULTILINE)
                for num, match in enumerate(matches, start=1):
                    sections_found.update({match.start(): sect})

    paragraphs_by_section = {section: [] for section in Section}
    sorted_section_pos = sorted(sections_found.keys())

    # If no regex for the header is defined, consider all text before the first section, if any, as header
    if Section.HEADER not in all_section_markers[Language.DE] and len(sorted_section_pos) > 0:
        paragraphs_by_section[Section.HEADER].append(decision[:sorted_section_pos[0]])   

    # Assign the corresponding part of the decision to its section 
    for i,match_start in enumerate(sorted_section_pos):
        actual_section = sections_found[match_start]
        from_ = match_start
        if i >= len(sorted_section_pos)-1: 
            # This is the last section, till end of decision
            to_ = len(decision)
        else:
            to_ = sorted_section_pos[i+1]
        paragraphs_by_section[actual_section].append(decision[from_:to_])

    # Validate the results
    error = True
    date = namespace['date']
    id = namespace['id']
    for defined_sections in all_section_markers[Language.DE]:
        if len(paragraphs_by_section[defined_sections]) != 0:
            error = False
            break
    if error == True:
        message = f'None of the section_markers gave any result. Date of the decision is {date} and id {id}'
        raise ValueError(message)

    return paragraphs_by_section

def NW_Gerichte(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [r'(Entscheid|Urteil|Zwischenentscheid|Beschluss|Abschreibungsentscheid|Abschreibungsverfügung) vom \d*\. (Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember) \d*',
                             r'\d*\.\/\d*\.\s(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s\d*'],
            Section.FACTS: [r'Sachverhalt:', r'Prozessgeschichte:', r'Nach Einsicht:'],
            Section.CONSIDERATIONS: [r'Erwägungen:'],
            Section.RULINGS: [r'Rechtsspruch:', r'(Demgemäss|Demnach) (beschliesst|erkennt|verfügt) (die|das) (Obergericht|Verfahrensleitung|Verwaltungsgericht|Prozessleitung)(:)*'],
            Section.FOOTER: [r'Stans\,\s\d*\.\s(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s\d*']
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

    paragraphs = get_pdf_paragraphs(decision)
    return associate_sections(paragraphs, section_markers, namespace)

def BE_ZivilStraf(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    all_section_markers = {
        Language.DE: {
            Section.FACTS: [r'^Sachverhalt:?\s*$', r'^Tatsachen$', 
                            r'^Prozessgeschichte und Eintreten$', 
                            r'^Ausgangslage$',
                            r'Sachverhalt und Beweiswürdigung'],
            Section.CONSIDERATIONS: [r'^Begründung:\s*$',r'Erwägung(en)?:?\s*$',
                                    r'^Entscheidungsgründe$', r'[iI]n Erwägung[:,]?\s*$',
                                    r'Aus den Erwägungen:',
                                    r'^Auszug aus den Erwägungen:'],
            Section.RULINGS: [ 
                            r'Strafkammer erkennt',     #Urteil
                            r'Strafkammer beschliesst', #Beschluss
                            r'Demgemäss erkennt d[\w]{2}', 
                            r'Appellationsgericht (\w+ )?(\(\w+\) )?erkennt', 
                            r'^und erkennt:$',
                            r'Das Gericht beschliesst:',
                            r'^Die Beschwerdekammer in Strafsachen beschliesst:$'],
            Section.FOOTER: [r'Hinweise:', r'Rechtsmittelbelehrung']
        }
    }

    valid_namespace(namespace, all_section_markers)

    sections_found = {}
    for lang in all_section_markers:
        for sect in all_section_markers[lang]:
            for reg in (all_section_markers[lang])[sect]:
                matches = re.finditer(reg, decision, re.MULTILINE)
                for num, match in enumerate(matches, start=1):
                    sections_found.update({match.start(): sect})

    paragraphs_by_section = {section: [] for section in Section}
    sorted_section_pos = sorted(sections_found.keys())
    if len(sorted_section_pos) == 0:
        raise ValueError(f"({namespace['id']}): No sections found at all. Please check! Here you have the url to the decision: {namespace['pdf_url']}")
    else:
        # If no regex for the header is defined, consider all text before the first section as header
        if Section.HEADER not in all_section_markers[Language.DE]:
            paragraphs_by_section[Section.HEADER].append(decision[:sorted_section_pos[0]])   

        # Assign the corresponding part of the decision to its section 
        for i,match_start in enumerate(sorted_section_pos):
            actual_section = sections_found[match_start]
            from_ = match_start
            if i >= len(sorted_section_pos)-1: 
                # This is the last section, till end of decision
                to_ = len(decision)
            else:
                to_ = sorted_section_pos[i+1]
            paragraphs_by_section[actual_section].append(decision[from_:to_])
    
    return paragraphs_by_section

def BE_Verwaltungsgericht(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """

    all_section_markers = {
        Language.DE: {
            Section.HEADER: [r'Verwaltungsgericht des Kantons Bern', r'Beschluss', r'SK\-*\s*Nr\.\s\d*/\d*',
                             r'\w*\-*\s*\d*\/*\s*\-*\d*\,\s(\w+\s)*\d{4}\s*\d*'],
            Section.FACTS: [r'Sachverhalt( und Erwägungen)*:', r'Regeste(:)*'],
            Section.CONSIDERATIONS: [r'Erwägungen:',
                                     r'(Der|Die|Das)\s\w+\s*(.+)\s*(e|E)rwäg(t|ung)(:)*(\,\s*dass)*(:)*'],
            Section.RULINGS: [r'Demnach entscheidet\s\w+\s*(.+)\s*:'],
            Section.FOOTER: [r'Rechtsmittelbelehrung']
        },

        Language.FR: {
            Section.HEADER: [r'Tribunal administratif du canton de Berne', r'\w+\-\d*\s\d*\,\s(\w+\s)*\d{4}',
                             r'Décision', r'\w*\s*\d*\s*\d*\,*\s*(\w+\.*\s)*\d{2}'],
            Section.FACTS: [r'En fait:'],
            Section.CONSIDERATIONS: [r'En droit:'],
            Section.RULINGS: [r'Par ces motifs:'],
            Section.FOOTER: [r'Voie de recours']
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

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)

    paragraphs = get_pdf_paragraphs(decision)
    return associate_sections(paragraphs, section_markers, namespace)

def BE_Steuerrekurs(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [r'STEUERREKURSKOMMISSION DES KANTONS BERN'],
            Section.FACTS: [r'(hat die Steuerrekurskommission den )*Akten entnommen:'],
            Section.CONSIDERATIONS: [r'Die Steuerrekurskommission zieht in Erwägung:'],
            Section.RULINGS: [r'Aus diesen Gründen wird erkannt:'],
            Section.FOOTER: [r'IM NAMEN DER STEUERREKURSKOMMISSION']
         },

        Language.FR: {
            Section.HEADER: [r'COMMISSION DES RECOURS'],
            Section.FACTS: [r'(La Commission des recours en matière fiscale )*constate en fait:'],
            Section.CONSIDERATIONS: [r'La Commission des recours en matière fiscale considère en droit:'],
            Section.RULINGS: [r'Par ces motifs, la Commission des recours en matière fiscale prononce:'],
            Section.FOOTER: [r'AU NOM DE LA COMMISSION DES RECOURS']
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

    paragraphs = get_pdf_paragraphs(decision)
    return associate_sections(paragraphs, section_markers, namespace)

def BS_Omni(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    # As soon as one of the strings in the list (regexes) is encountered we switch to the corresponding section (key)
    # (?:C|c) is much faster for case insensitivity than [Cc] or (?i)c
    all_section_markers = {
        Language.DE: {
            Section.FACTS: [r'^Sachverhalt:?\s*$', r'^Tatsachen$'],
            Section.CONSIDERATIONS: [r'^Begründung:\s*$',r'Erwägung(en)?:?\s*$',r'^Entscheidungsgründe$', r'[iI]n Erwägung[:,]?\s*$'],
            Section.RULINGS: [r'Demgemäss erkennt d[\w]{2}', r'erkennt d[\w]{2} [A-Z]\w+:', r'Appellationsgericht (\w+ )?(\(\w+\) )?erkennt', r'^und erkennt:$', r'erkennt:\s*$'],
            Section.FOOTER: [r'^Rechtsmittelbelehrung$',
                             r'AUFSICHTSKOMMISSION', r'APPELLATIONSGERICHT']
        }
    }
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)

    divs = decision.find_all(
        "div", class_=['WordSection1', 'Section1', 'WordSection2'])
    paragraphs = get_paragraphs(divs)
    return associate_sections(paragraphs, section_markers, namespace)

def VD_Omni(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    # As soon as one of the strings in the list (regexes) is encountered we switch to the corresponding section (key)
    # (?:C|c) is much faster for case insensitivity than [Cc] or (?i)c
    all_section_markers = {
              Language.FR: {
            Section.FACTS: [r'^[L,l]a Cour de droit administratif et public\s?[,:]?\s?$', r'Faits\s?:', r'[E,e]n fait et en droit', r'(?:V|v)u\s?:', r'A.-', r'[V,v]u les faits suivants', r'constate en fait(\s*)?:?', r'^[E,e]n fait\s?:?$', r'^[V,v]u en fait\s?:?'],
            Section.CONSIDERATIONS: [r'Considérant en (?:fait et en )?droit\s?:?', r'(?:C|c)onsidérant(s?)\s?:?$',
                                     r'^considère\s?:?$', r'^(et)?\s?[C,c]onsidère en droit\s?:?$', r'[E,e]n droit\s?[:,]?\s?$'],
            Section.RULINGS: [r'^prononce\s?:', r'^[P,p]ar ces? motifs?\s?[,:]?\s?', r'^ordonne\s?:'],
            Section.FOOTER: [
                r'^([^\s]*)?\w*,\s(le\s?)?((\d?\d)|\d\s?(er|re|e)|premier|première|deuxième|troisième)\s?(?:janv|févr|mars|avr|mai|juin|juill|août|sept|oct|nov|déc).{0,10}\d?\d?\d\d\s?([^\s]*)?\w*$',
                r'Au nom de la Cour', r'^Lausanne, le$'
            ]
        },
    }
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)

    divs = decision.find_all(
        "div", class_=['WordSection1', 'Section1', 'WordSection2'])
    paragraphs = get_paragraphs(divs)
    return associate_sections(paragraphs, section_markers, namespace)


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

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)

    divs = decision.find_all("div", class_="content")
    # we expect maximally two divs with class content
    assert len(divs) <= 2

    paragraphs = get_paragraphs(decision)
    return associate_sections(paragraphs, section_markers, namespace)

def get_paragraphs(divs):
    # """
    # Get Paragraphs in the decision
    # :param divs:
    # :return:
    # """  
    paragraphs = []
    heading, paragraph = None, None
    for div in divs:
        for element in div:
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
        element = element.replace('  ',' ')
        paragraph = clean_text(element)
        if paragraph not in ['', ' ', None]:  # discard empty paragraphs
            paragraphs.append(paragraph)
    return paragraphs

def valid_namespace(namespace: dict, all_section_markers):
    """
    Check if the section markers have been implemented for a given language
    :param namespace:               the namespace containing some metadata of the court decision
    :param all_section_markers:     the section markers of a decision
    """
    if namespace['language'] not in all_section_markers:
        message = f"This function is only implemented for the languages {list(all_section_markers.keys())} so far."
        raise ValueError(message)   

def prepare_section_markers(all_section_markers, namespace: dict) -> Dict[Section, str]: 
    """
    Join and normalize the section markers
    :param all_section_markers:     the section markers of a decision
    :param namespace:               the namespace containing some metadata of the court decision
    :return:                        a Dict of the Section and the section markers
    """
    section_markers = all_section_markers[namespace['language']]
    section_markers = dict(
        map(lambda kv: (kv[0], '|'.join(kv[1])), section_markers.items()))
    for section, regexes in section_markers.items():
        section_markers[section] = unicodedata.normalize('NFC', regexes)
    return section_markers

def associate_sections(paragraphs: List[str], section_markers, namespace: dict, sections: List[Section] = list(Section)):
    """
    Associate sections to paragraphs
    :param paragraphs:      list of paragraphs
    :param section_markers: dict of section markers
    :param namespace:       dict of namespace
    :param sections:        if some sections are not present in the court, pass a list with the missing section excluded
    """
    paragraphs_by_section = { section: [] for section in sections }

    # assert that for every passed section a section_marker is present, the header is included by default
    assert set(sections) == set(section_markers.keys()).union(set([Section.HEADER])), \
        f"Missing section marker: {set(sections) - set(section_markers.keys()).union(set([Section.HEADER]))}"
    current_section = Section.HEADER
    for paragraph in paragraphs:
        # update the current section if it changed
        current_section = update_section(current_section, paragraph, section_markers, sections)
        # add paragraph to the list of paragraphs
        paragraphs_by_section[current_section].append(paragraph)
    if current_section != Section.FOOTER:
        # change the message depending on whether there's a url
        if namespace['html_url']:
            message = f"({namespace['id']}): We got stuck at section {current_section}. Please check! " \
                  f"Here you have the url to the decision: {namespace['html_url']}"
        elif 'pdf_url' in namespace and namespace['pdf_url']:
            message = f"({namespace['id']}): We got stuck at section {current_section}. Please check! " \
                  f"Here is the url to the decision: {namespace['pdf_url']}"
        else:
            message = f"({namespace['id']}): We got stuck at section {current_section}. Please check! " \
                  f"Here is the date of the decision: {namespace['date']}"
        get_logger(__name__).warning(message)
    return paragraphs_by_section

def update_section(current_section: Section, paragraph: str, section_markers, sections: List[Section]) -> Section:
    """
    Update the current section if it changed
    :param current_section: the current section
    :param paragraph:       the current paragraph
    :param section_markers: dict of section markers
    :param sections:        if some sections are not present in the court, pass a list with the missing section excluded
    :return:                the updated section
    """
    paragraph = unicodedata.normalize('NFC', paragraph)  # if we don't do this, we get weird matching behaviour
    if current_section == Section.FOOTER:
        return current_section  # we made it to the end, hooray!
    next_section_index = sections.index(current_section) + 1
    # consider all following sections
    next_sections = sections[next_section_index:]
    for next_section in next_sections:
        marker = section_markers[next_section]
        if re.search(marker, paragraph):
            return next_section  # change to the next section
    return current_section  # stay at the old section

# This needs special care
# def CH_BGE(decision: Any, namespace: dict) -> Optional[dict]:
#    return CH_BGer(decision, namespace)

def ZG_Verwaltungsgericht(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    Split a decision of the Verwaltungsgericht of Zug into several named sections
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    # As soon as one of the strings in the list (regexes) is encountered we switch to the corresponding section (key)
    all_section_markers = {
        Language.DE: {
            # "header" has no markers!
            Section.FACTS: [r'wird Folgendes festgestellt:', r'wird nach Einsicht in', r'^A\.\s', r'^A\.a\)\s'],
            Section.CONSIDERATIONS: [r'(Der|Die|Das) \w+ erwägt:', r'und in Erwägung, dass'],
            Section.RULINGS: [r'Demnach erkennt', r'Folgendes verfügt', r'(Der|Die|Das) \w+ verfügt:', r'Demnach wird verfügt:', r'Demnach wird erkannt'],
            Section.FOOTER: [r'^[\s]*Zug,( den| vom)?\s\d?\d\.?\s?(?:Jan(?:uar)?|Feb(?:ruar)?|Mär(?:z)?|Apr(?:il)?|Mai|Jun(?:i)?|Jul(?:i)?|Aug(?:ust)?|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?|Dez(?:ember)?)\s\d{4}']
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

    # This court sometimes uses newlines to separate names of people. 
    # To deal with that, this loop inserts a comma if a new line starts with lic. iur. to separate names.
    lines = []
    lines = decision.split('\n')
    for idx, line in enumerate(lines):
        if 'lic. iur.' in line:
            line = re.sub(r'^lic\. iur\.', ', lic. iur.', line)
            lines[idx] = line
    decision = '\n'.join(map(str, lines))

    paragraphs = get_pdf_paragraphs(decision)
    return associate_sections(paragraphs, section_markers, namespace)


def ZH_Baurekurs(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    Split a decision of the Baurekursgericht of Zurich into several named sections
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    # As soon as one of the strings in the list (regexes) is encountered we switch to the corresponding section (key)
    all_section_markers = {
        Language.DE: {
            # "header" has no markers!
            Section.FACTS: [r'hat sich ergeben', r'Gegenstand des Rekursverfahrens'],
            Section.CONSIDERATIONS: [r'Es kommt in Betracht', r'Aus den Erwägungen'],
            Section.RULINGS: [r'(Zusammengefasst|Zusammenfassend) (ist|sind)', r'(Zusammengefasst|Zusammenfassend) ergibt sich', r'Der Rekurs ist nach', r'Gesamthaft ist der Rekurs', r'Dies führt zur (Aufhebung|Abweisung|Gutheissung|teilweisen)'],
            # there are generally no footers
            Section.FOOTER: [r'Im Namen des Baurekursgerichts']
        },
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

    paragraphs = get_pdf_paragraphs(decision)
    return associate_sections(paragraphs, section_markers, namespace)



def ZH_Obergericht(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    Split a decision of the Obergericht of Zurich into several named sections
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    # As soon as one of the strings in the list (regexes) is encountered we switch to the corresponding section (key)
    all_section_markers = {
        Language.DE: {
            # "header" has no markers!
            Section.FACTS: [r'^[\s]*betreffend(\s|$)', r'Sachverhalt:'],
            Section.CONSIDERATIONS: [r'(?:A|a)us den Erwägungen ', r'Erwägungen:', r'^[\s]*Erwägungen[\s]*$', r'Das (Einzelgericht|Gericht) erwägt', r'Das (Einzelgericht|Gericht) zieht in (Erwägung|Betracht)'],
            Section.RULINGS: [r'^[\s]*Es wird (erkannt|beschlossen|verfügt):', r'^[\s]*wird beschlossen:[\s]*$', r'Das (Einzelgericht|Gericht) (erkennt|beschliesst):', r'(Sodann|Demnach|Demgemäss) beschliesst das Gericht:'],
            Section.FOOTER: [
                r'^[\s]*Zürich,( den| vom)?\s\d?\d\.?\s?(?:Jan(?:uar)?|Feb(?:ruar)?|Mär(?:z)?|Apr(?:il)?|Mai|Jun(?:i)?|Jul(?:i)?|Aug(?:ust)?|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?|Dez(?:ember)?)\s\d{4}([\s]*$)', r'OBERGERICHT DES KANTONS ZÜRICH']
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

    paragraphs = get_pdf_paragraphs(decision)
    return associate_sections(paragraphs, section_markers, namespace)


def ZH_Sozialversicherungsgericht(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    Split a decision of the Sozialversicherungsgericht of Zurich into several named sections
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    # As soon as one of the strings in the list (regexes) is encountered we switch to the corresponding section (key)
    all_section_markers = {
        Language.DE: {
            # "header" has no markers!
            Section.FACTS: [r'Sachverhalt:', r'^[\s]*Sachverhalt[\s]*$', r'Unter Hinweis darauf,'],
            Section.CONSIDERATIONS: [r'in Erwägung,', r'zieht in Erwägung:', r'Erwägungen:'],
            Section.RULINGS: [r'Das Gericht (erkennt|beschliesst|verfügt):', r'(Der|Die) Einzelrichter(in)? (erkennt|beschliesst|verfügt):', r'(beschliesst|erkennt) das Gericht:', r'und erkennt sodann:', r'(Der|Die) Referent(in)? (erkennt|beschliesst|verfügt):'],
            # this court only sometimes has a footer
            Section.FOOTER: [r'Im Namen des Sozialversicherungsgerichts', r'^[\s]*Sozialversicherungsgericht des Kantons Zürich[\s]*$']
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

    # This should be the div closest to the content:
    content = decision.find("div", id="view:_id1:inputRichText1")
    multiple_results = False
    # Sometimes there is no content:
    if len(content.contents) == 0:
        return
    # Ideally, the content is directly below the above div, but if not:
    if len(content.contents) <= 5:
        # The main content should have more than 5 children:
        div = content.find('div')
        if div and len(div.contents) >= 5:
            # There's a div with enough children to be the main content:
            # But maybe there's more than one:
            content_list = [tag for tag in content.find_all("div") if len(tag.contents) > 1]
            if content_list and len(content_list) > 1 and not content.find_all(class_="domino-par--indent"):
                multiple_results = True
            elif content.find_all(class_="domino-par--indent"):
                content_list = [tag for tag in content.find_all("div", class_="domino-par--indent", recursive=False) if len(tag.contents) > 1]
                multiple_results = True
            # If there's only one:
            else:
                content = div
                assert len(content.contents) >= 5
        elif not div:
            # If the div doesn't exist, there should be a ul directly below the id:
            content = content.find("ul", class_="domino-par--indent", recursive=False)
            assert len(content) >= 5
        elif div and len(div.contents) == 1 and not div.find_all(class_="domino-par--indent"):
            # Possibly there's a div with the content directly below the div
            div2 = div.find('div', recursive=False)
            if div2 and len(div2.contents) >= 5:
                content = div2
            else:
                pass
        elif div and len(div.contents) < 5:
            # The relevant content has class 'domino-par--indent' and the following style:
            content_list = [tag for tag in content.find_all(class_="domino-par--indent", attrs={'style':'padding-left: 62pt'}) if len(tag.contents) > 1]
            if len(content_list) > 0:
                multiple_results = True
            else:
                # Sometimes the relevant content has this style
                content_list = [tag for tag in content.find_all(class_="domino-par--indent", attrs={'style':'padding-left: 85pt'}) if len(tag.contents) > 1]
                if len(content_list) > 0:
                    multiple_results = True
                else:
                    # Sometimes there is no style but this is less precise
                    content_list = [tag for tag in content.find_all(class_="domino-par--indent") if len(tag.contents) > 1]
                    multiple_results = True

    def get_paragraphs(content):
        """
        Get the paragraphs from a piece of html content
        :param soup:    the content parsed by bs4
        :return:        a list of paragraphs
        """
        paragraphs = []
        heading, paragraph = None, None
        for element in content:
            if isinstance(element, bs4.element.Tag):
                text = str(element.string)
                # This is a hack to also get tags which contain other tags such as links to BGEs
                if text.strip() == 'None':
                    # replace br tags with spaces and insert spaces before div end tags
                    # without this, words might get stuck together
                    html_string = str(element)
                    html_string = html_string.replace('<br>', ' ').replace('<br/>', ' ').replace('<br />', ' ').replace('</div>', ' </div>')
                    element = bs4.BeautifulSoup(html_string, 'html.parser')
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
                if paragraph not in ['', ' ', None]:  # only clean and append non-empty paragraphs
                    paragraph = clean_text(paragraph)
                    paragraphs.append(paragraph)
        return paragraphs

    if content:
        paragraphs = []
        if multiple_results:
            for el in content_list:
                paragraphs += get_paragraphs(el)
        else:
            paragraphs = get_paragraphs(content)
        return associate_sections(paragraphs, section_markers, namespace)
    else:
        return




def ZH_Steuerrekurs(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    Split a decision of the Steuerrekursgericht of Zurich into several named sections
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    # As soon as one of the strings in the list (regexes) is encountered we switch to the corresponding section (key)
    all_section_markers = {
        Language.DE: {
            # "header" has no markers!
            Section.FACTS: [r'hat sich ergeben:'],
            Section.CONSIDERATIONS: [r'zieht in Erwägung:', r'sowie in der Erwägung'],
            Section.RULINGS: [r'Demgemäss (erkennt|beschliesst|verfügt)', r'beschliesst die Rekurskommission', r'verfügt der Einzelrichter', r'verfügt die Einzelrichterin'],
            # there is generally no footer
            Section.FOOTER: [
                r'Im Namen des Steuerrekursgerichts']
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

    paragraphs = get_pdf_paragraphs(decision)
    return associate_sections(paragraphs, section_markers, namespace)



def ZH_Verwaltungsgericht(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    Split a decision of the Verwaltungsgericht of Zurich into several named sections
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    # As soon as one of the strings in the list (regexes) is encountered we switch to the corresponding section (key)
    all_section_markers = {
        Language.DE: {
            # "header" has no markers!
            Section.FACTS: [r'hat sich ergeben:', r'^\s*I\.\s+A\.\s*', r'^\s*I\.\s+(&nbsp;)?$', r'^\s*I\.\s[A-Z]+', r'nach Einsichtnahme in', r'Sachverhalt[:]?[\s]*$'],
            Section.CONSIDERATIONS: [r'erwägt:', r'zieht in (Erwägung|Betracht)', r'zieht (der Einzelrichter|die Einzelrichterin) in Erwägung', r'in Erwägung, dass', r'(?:A|a)us den Erwägungen', r'hat erwogen:'],
            Section.RULINGS: [r'(Demgemäss|Demnach|Dementsprechend|Demmäss) (erkennt|erkannt|beschliesst|entscheidet|verfügt)', r'Das Verwaltungsgericht entscheidet', r'(Die Kammer|Der Einzelrichter|Die Einzelrichterin) (erkennt|entscheidet|beschliesst|hat beschlossen)', r'Demgemäss[\s|(&nbsp;)]*die Kammer:', r'Der Abteilungspräsident verfügt:', r'^[\s]*verfügt[:]?[\s]*$', r'^[\s]*entschieden:[\s]*$', r'^[\s]*und (entscheidet|erkennt):[\s]*$'],
            # this court generally has no footer
            Section.FOOTER: [r'Im Namen des Verwaltungsgerichts']
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
        :param soup:    the decision parsed by bs4 
        :return:        a list of paragraphs
        """
        # sometimes the div with the content is called WordSection1
        divs = soup.find_all("div", class_="WordSection1")
        # sometimes the div with the content is called Section1
        if len(divs) == 0:
            divs = soup.find_all("div", class_="Section1")
        # we expect maximally two divs with class WordSection1
        assert (len(divs) <= 2),"Found more than two divs with class WordSection1"
        assert (len(divs) > 0),"Found no div, " + str(namespace['html_url'])

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
                if paragraph not in ['', ' ', None]:  # only clean and append non-empty paragraphs
                    paragraph = clean_text(paragraph)
                    paragraphs.append(paragraph)
        return paragraphs

    paragraphs = get_paragraphs(decision)
    return associate_sections(paragraphs, section_markers, namespace)

# returns dictionary with section names as keys and lists of paragraphs as values
def BE_BVD(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:

    # split sections by given regex, compile regex to cache them
    regexes = {
        Language.DE: re.compile(r'(.*?)(Sachverhalt(?:\n  \n|\n\n|\n \n| \n\n).*?)(Erwägungen(?: \n\n|\n\n).*?)(Entscheid(?:\n\n| \n\n1).*?)((?:Eröffnung(?:\n\n|\n-)|[Zz]u eröffnen:).*)', re.DOTALL),
        Language.FR: re.compile(r'(.*?)(Faits\n\n.*?)(Considérants\n\n.*?)(Décision\n\n.*?)(Notification\n\n|A notifier:\n.*)', re.DOTALL)
    }

    try:
        regex = regexes[namespace['language']]
    except KeyError:
        message = f"This function is only implemented for the languages {list(regexes.keys())}."
        raise ValueError(message)
    
    match = re.search(regex, decision)
    matches = []

    if match is None:
        # if sachverhalt and erwägungen are in the same section, add them to both sections
        if re.search('Sachverhalt und Erwägungen\n', decision, re.M):
            edge_case_regex = re.compile(r'(.*?)(Sachverhalt und Erwägungen(?: \n\n|\n\n).*?)(Entscheid(?:\n\n| \n\n1).*?)((?:Eröffnung(?:\n\n|\n-)|[Zz]u eröffnen:).*)', re.DOTALL)
            match = re.search(edge_case_regex, decision)
            if match is None:
                # TODO: change to pdf_url once supported
                raise ValueError(f"Could not find sections for decision {namespace['id']}")
            
            matches = list(match.groups())
            # add sachverhalt and erwägungen to both sections
            matches = [matches[0], matches[1]] + matches[1:]
        else:
            raise ValueError(f"Could not find sections for decision{namespace['id']}")
    else:
        matches = list(match.groups())
    
    # split paragraphs
    sections = {}
    for section, section_text in zip(list(Section), matches):

        split = re.split('(\\n\d\. \w+\\n)', section_text)
        # paragraphs are now split into title, paragraph header (e.g. '\n1. '), paragraph text
        title = split[0]
        # join header and text pairs back together (1+2, 3+4, 5+6, ...) if we found multiple (>2) paragraphs
        paired = []
        if len(split) > 2:
            paired = [split[i] + split[i+1] for i in range(1, len(split) -1, 2)]
        else:
            paired = list(''.join(split[1:]))

        sections[section] = [title] + paired
    
    return sections

def BE_ZivilStraf(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
       
    Remarks:
    * This court does not have a facts section (few edge cases), but concatenates the facts with the considerations. For now both are
      added to the considerations section, but it might make sense to create a new type of section which unions the two.
    * Accuracy could still be improved, but the decisions which are not fully matched are edge-cases with typos or weird
      pdf parsing errors.
    * About the poor performance on the footer section in german decisions: The facts and considerations
      section often contains a summary with the exact same keywords used to detect the footer:
          '\n\nWeiter wird verfügt:\n\n'
      This could be solved but the possibilites are not very straight-forward. Some options are:
        - estimate on where in the document the footer would be and try to extract it from that text part
        - use the second match to split it apart. As the association between section markers and sections
          is done in a helper method, this parser would have to be rewritten from scratch as the helper
          cannot be updated to support this case.
    * The rulings section in the german version has bad accuracy because many decision do not have a rulings section.
      examples of the last paragraph of the considerations section:
      - "Aus den Darlegungen erhellt, dass die Konkursandrohung in der Betreibung Nr. 123
        des Betreibungs- und Konkursamtes B, Dienststelle P., nicht nichtig ist."
      - "Nach dem Gesagten ist auf die Beschwerde nicht einzutreten."
      - "Vor diesem Hintergrund ist das Vorgehen der Dienststelle unter den dargelegten
        Umständen nicht zu beanstanden und die vorliegende Beschwerde abzuweisen."
      This is resolved by using the last paragraph of the considerations section as the rulings section.
    * The problem with the footer detection is the same for the rulings, as they are mentioned with the same keywords in
      the summary of the considerations as well.
    """

    markers = {
        Language.DE: {
            # "header" has no markers!
            # "facts" are not present either in this court, leave them out
            Section.CONSIDERATIONS: [r'^Erwägungen:|^Erwägungen$', r'Auszug aus den Erwägungen', r'Formelles$', '^Sachverhalt(?: |:)'],
            Section.RULINGS: [r'^Die (?:Aufsichtsbehörde|Kammer) entscheidet:', r'(?:^|\. )Dispositiv',
                              r'^Der Instrkutionsrichter entscheidet:', r'^Strafkammer erkennt:',
                              r'^Die Beschwerdekammer in Strafsachen (?:beschliesst|hat beschlossen):', r'^Das Gericht beschliesst:',
                              r'^Die Verfahrensleitung verfügt:', r'^Der Vizepräsident entscheidet:',
                              r'^Das Handelsgericht entscheidet:', r'^Die \d. Strafkammer beschliesst:'],
            # "Weiter wird verfügt:" often causes problems with summarys in the considerations section, leave it out
            Section.FOOTER: [r'^Zu eröffnen:', r'\d\. Zu eröffnen:', r'^Schriftlich zu eröffnen:$',
                             r'^Rechtsmittelbelehrung', r'^Hinweis:'] # r'^Weiter wird verfügt:'
        },
        Language.FR: {
            # "header" has no markers!
            # "facts" are not present either in this court, leave them out
            Section.CONSIDERATIONS: [r'^Considérants(?: :|:)?', r'^Extrait des (?:considérations|considérants)(?: :|:)'],
            Section.RULINGS: [r'^La Chambre de recours pénale décide(?: :|:)', r'^Dispositif'],
            Section.FOOTER: [r'A notifier(?: :|:)', r'Le présent (?:jugement|dispositif) est à notifier(?: :|:)',
                             r'Le présent jugement est à notifier par écrit(?: :|:)']
        }
    }

    if namespace['language'] not in markers:
        message = f"This function is only implemented for the languages {list(markers.keys())} so far."
        raise ValueError(message)
    
    section_markers = markers[namespace['language']]

    # combine multiple regex into one for each section due to performance reasons
    section_markers = dict(map(lambda kv: (kv[0], '|'.join(kv[1])), section_markers.items()))

    # normalize strings to avoid problems with umlauts
    for section, regexes in section_markers.items():
        section_markers[section] = unicodedata.normalize('NFC', regexes)

    def get_paragraphs(soup):
        """
        Get Paragraphs in the decision
        :param soup: the string extracted of the pdf
        :return: a list of paragraphs
        """
        paragraphs = []
        # remove spaces between two line breaks
        soup = re.sub('\\n +\\n', '\\n\\n', soup)
        # split the lines when there are two line breaks
        lines = soup.split('\n\n')
        for element in lines:
            element = element.replace('  ',' ')
            paragraph = clean_text(element)
            if paragraph not in ['', ' ', None]:
                paragraphs.append(paragraph)
        return paragraphs

    paragraphs = get_paragraphs(decision)

    # pass custom sections without facts
    sections = associate_sections(paragraphs, section_markers, namespace, list(Section.without_facts()))

    # regularly happens that the decision is within the CONSIDERATIONS section, so if no rulings are found by the
    # section_markers we try to extract the rulings from the considerations section instead
    if sections[Section.RULINGS] == [] and sections[Section.CONSIDERATIONS] != []:
        # got no rulings, use the chance that the ruling is in the considerations section, traverse backwards to find it
        for index, paragraph in enumerate(reversed(sections[Section.CONSIDERATIONS])):
            # make sure the paragraph contains some ruling keywords
            keywords = r"abzuweisen|Abweisung der Beschwerde|gutzuheissen|Beschwerde gutgeheissen|rechtsgenüglich begründet|" \
                       r"Beschwerde [\w\s]* als begründet\.|obsiegend"
            if re.findall(keywords, paragraph):
                # if res contains some ruling keywords it is the decision, remove it from considerations, add it to rulings
                # add everything after it to the ruling as well
                sections[Section.RULINGS] = sections[Section.CONSIDERATIONS][index:]
                sections[Section.CONSIDERATIONS] = sections[Section.CONSIDERATIONS][:index]
                break
    
    return sections

def CH_BPatG(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    Remark: This court does not have a facts section, and some don't have a footer.
    """
    markers = {
        Language.DE: {
            # Section.FACTS: [], # no facts in this court
            Section.CONSIDERATIONS: [r'^(?:Das Bundespatentgericht|(?:Der|Das) Präsident|Die Gerichtsleitung|Das Gericht|Der (?:Einzelrichter|Instruktionsrichter))' \
                                      r' zieht in Erwägung(?:,|:)',
                                     r'Der Präsident erwägt:', r'Aus(?:|zug aus) den Erwägungen:', r'Sachverhalt:'],
            Section.RULINGS: [r'(?:Der Instruktionsrichter|Das Bundespatentgericht|(?:Das|Der) Präsident) (?:erkennt|verfügt|beschliesst)(?:,|:)',
                              r'Die Gerichtsleitung beschliesst:', r'Der Einzelrichter erkennt:'],
            Section.FOOTER: [r'Rechtsmittelbelehrung:', r'Dieser Entscheid geht an:']     
        },
        Language.FR: {
            # Section.FACTS: [], # no facts in this court
            Section.CONSIDERATIONS: [r'Le Tribunal fédéral des brevets considère(?: :|:|,)', r'Le [pP]résident considère(?: :|:|,)'],
            Section.RULINGS: [r'Le Tribunal fédéral des brevets décide:', r'Le [pP]résident (décide|reconnaît):'],
            Section.FOOTER: [r'Voies de droit:']
        },
        Language.IT: {
            # Section.FACTS: [], # no facts in this court
            Section.CONSIDERATIONS: [r'Considerando in fatto e in diritto:'],
            Section.RULINGS: [r'Per questi motivi, il giudice unico pronuncia:'],
            Section.FOOTER: [r'Rimedi giuridici:']
        }
    }

    if namespace['language'] not in markers:
        message = f"This function is only implemented for the languages {list(markers.keys())}, not {namespace['language']}."
        raise ValueError(message)
    
    section_markers = markers[namespace['language']]

    # combine multiple regex into one for each section due to performance reasons
    section_markers = dict(map(lambda kv: (kv[0], '|'.join(kv[1])), section_markers.items()))

    # normalize strings to avoid problems with umlauts
    for section, regexes in section_markers.items():
        section_markers[section] = unicodedata.normalize('NFC', regexes)
    
    if namespace['language'] == Language.DE:
        # remove the page numbers, they are not relevant for the decisions
        decision = re.sub(r'Seite \d', '', decision)

    def get_paragraphs(soup):
        """
        Get Paragraphs in the decision
        :param soup: the string extracted of the pdf
        :return: a list of paragraphs
        """
        paragraphs = []
        # remove spaces between two line breaks
        soup = re.sub('\\n +\\n', '\\n\\n', soup)
        # split the lines when there are two line breaks
        lines = soup.split('\n\n')
        for element in lines:
            element = element.replace('  ',' ')
            paragraph = clean_text(element)
            if paragraph not in ['', ' ', None]:
                paragraphs.append(paragraph)
        return paragraphs

    paragraphs = get_paragraphs(decision)

    # pass custom sections without facts
    return associate_sections(paragraphs, section_markers, namespace, list(Section.without_facts()))
