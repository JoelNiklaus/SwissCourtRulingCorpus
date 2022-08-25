from logging import exception
import unicodedata
from typing import Optional, List, Dict, Union
import sys

import bs4
import re

from prometheus_client import Enum
from sqlalchemy import false, true

from scrc.enums.language import Language
from scrc.enums.section import Section
from scrc.utils.main_utils import clean_text
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_paragraphs_unified

from scrc.preprocessors.extractors.spider_specific.judgment_extracting_functions import all_judgment_markers


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

    # This is how a "standard" section splitting function looks like.
    # First specify the markers where to split, then prepare them by joining and normalizing them.
    # Then get the paragraphs and loop through them with the markers using the associate_sections function.
    """ all_section_markers = {
        Language.DE: {
            Section.FACTS: [r'Tatbestand', r'Sachverhalt'],
            Section.CONSIDERATIONS: [r"Erwägung"],
            Section.RULINGS: [r"Demnach erkennt", r"Demnach beschliesst", r"Demnach wird beschlossen", r"Demnach wird verfügt", r"Dispositiv"],
            Section.FOOTER: [r""]
        }
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)

    divs = decision.find_all(
        "div", class_="content")
    paragraphs = get_paragraphs(divs)

    return associate_sections(paragraphs, section_markers, namespace) """



def GE_Gerichte(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.FR: {
            Section.HEADER: [],
            Section.FACTS: [r'EN FAIT', r'en fait'],
            Section.CONSIDERATIONS: [r'EN DROIT', 'en droit'],
            Section.RULINGS: [r'PAR CES MOTIFS', r'LA CHAMBRE ADMINISTRATIVE'],
            Section.FOOTER: [r'La [g,G]reffière', r'la [G,g]reffière', r'Siégeant', r'Voie de recours', r'Le recours doit être', r'Le [G,g]reffier', r'Le [P,p]résident']
        },
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [r'Tatbestand', r'Sachverhalt'],
            Section.CONSIDERATIONS: [r"Erwägung"],
            Section.RULINGS: [r"Demnach erkennt", r"Demnach beschliesst", r"Demnach wird beschlossen",
                              r"Demnach wird verfügt", r"Dispositiv"],
            Section.FOOTER: [r'Rechtsmittellehre']
        },
        Language.EN: {
            Section.HEADER: [],
            Section.FACTS: [],
            Section.CONSIDERATIONS: [],
            Section.RULINGS: [],
            Section.FOOTER: []
        },
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace)

def BE_Anwaltsaufsicht(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.FACTS: [r'Sachverhalt'],
            Section.CONSIDERATIONS: [r'Erwägungen'],
            Section.RULINGS: [r'Die Anwaltsaufsichtsbehörde entscheidet:'],
            Section.FOOTER: [r'Der Präsident:', r'Hinweis: Dieser Entscheid ist rechtskräftig', r'Die Präsidentin', r'Rechtsmittelbelehrung']
        },
        Language.FR: {
            Section.FACTS: [r'Sachverhalt'],
            Section.CONSIDERATIONS: [r'Considérants:'],
            Section.RULINGS: [r'Pour ces motifs,'],
            Section.FOOTER: [r'Voies de recours']
        }
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace)


def BE_Weitere(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.FACTS: [r'Sachverhalt'],
            Section.CONSIDERATIONS: [r'Erwägung', r'Erwägungen', r'erwogen', r'Ausgangslage$'],
            Section.RULINGS: [r'entscheidet', r'wird erkannt', r'erkannt :', r'erkannt:', r'III. Entscheid', r'[1-9] Entscheid'],
            Section.FOOTER: [r'^Rechtsmittelbelehrung']
        },
        Language.FR: {
            Section.FACTS: [r'Faits', r'de fait', r'En fait:', r'les faits'],
            Section.CONSIDERATIONS: [r'considère:', r'Considérants', r'En droit', r'Considérations sur le fond'],
            Section.RULINGS: [r'Pour ces motifs', r'Frais de procédure', r'Par ces motifs'],
            Section.FOOTER: [r'Voies de recours', r'Indication des voies de droit']
        }
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace)


def AR_Gerichte(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.FACTS: [r'Sachverhalt'],
            Section.CONSIDERATIONS: [r'Erwägungen'],
            Section.RULINGS: [r'erkennt', r'beschliesst'],
            Section.FOOTER: [r'La [g,G]reffière']
        }
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    if isinstance(decision, str):
        paragraphs = get_pdf_paragraphs(decision)
    else:
        paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace)


def BE_Steuerrekurs(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.FACTS: [r'den Akten entnommen'],
            Section.CONSIDERATIONS: [r'Die Steuerrekurskommission zieht in Erwägung', 'en droit'],
            Section.RULINGS: [r'Aus diesen Gründen wird erkannt:'],
            Section.FOOTER: [r'IM NAMEN DER STEUERREKURSKOMMISSION DES KANTONS BERN']
        },
        Language.FR: {
            Section.FACTS: [r'constate en fait'],
            Section.CONSIDERATIONS: [r'considère en droit'],
            Section.RULINGS: [r'Par ces motifs'],
            Section.FOOTER: [r'AU NOM DE LA COMMISSION DES RECOURS EN MATIERE FISCALE DU CANTON DE BERNE', r'AU NOM DE LA COMMISSION DES RECOURS EN MATIÈRE FISCALE DU CANTON DE BERNE']
        }
    }
    

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_pdf_paragraphs(decision)

    return associate_sections(paragraphs, section_markers, namespace)

def GL_Omni(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [r'in Sachen', r'Sachverhalt'],
            Section.CONSIDERATIONS: [r'Erwägung', r'Erwägungen', r'Betracht:?$'],
            Section.RULINGS: [r'[D,d]emgemäss erkennt', r'erkennt sodann', r'Gericht[\s]*erkennt', r'Gericht beschliesst', r'zieht in Betracht', r'verfügt:?$', r'[D,d]emgemäss beschliesst', r'beschliesst:?$', r'erkennt:?$'],
            Section.FOOTER: [r'^Rechtsmittelbelehrung:?$']
        }
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace)

def BL_Gerichte(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [r'Sachverhalt'],
            Section.CONSIDERATIONS: [r'zieht i n E r w ä g u n g', r'Erwägungen', r'in Erwägung:'],
            Section.RULINGS: [r'Demgemäss wird e r k a n n t', r'Demgemäss w i r d e r k a n n t', r'Demnach wird erkannt', r'Demgemäss wird erkannt', r'Demnach erkennt das Steuergericht:', r'Demgemäss erkennt das Steuergericht:', r'wird erkannt:', r'Es wird erkannt:'],
            Section.FOOTER: [r'Rechtsmittelbelehrung', r'^Präsidentin$', r'^Präsident$', r'^Gerichtsschreiberin$']
        }
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace)

def AG_Gerichte(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [ r'^Sachverhalt', r'entnimmt den Akten:'],
            Section.CONSIDERATIONS: [r'in Erwägung:', r'Aus den Erwägungen', r'^Erwägungen$'],
            Section.RULINGS: [r'erkennt:?$', r'beschliesst:?$', r'entscheidet:?$'],
            Section.FOOTER: [r'^Rechtsmittelbelehrung', r'Der Vizepräsident: Der Gerichtsschreiber:', r'Der Präsident: Die Gerichtsschreiberin:']
        },
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace)

def AG_Weitere(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [ r'^Sachverhalt', r'entnimmt den Akten:'],
            Section.CONSIDERATIONS: [r'in Erwägung:', r'Aus den Erwägungen'],
            Section.RULINGS: [r'erkennt:?$', r'beschliesst:?$', r'entscheidet:?$'],
            Section.FOOTER: [r'^Rechtsmittelbelehrung', r'Der Vizepräsident: Der Gerichtsschreiber:', r'Der Präsident: Die Gerichtsschreiberin:']
        },
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace)

def CH_WEKO(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [r'Sachverhalt$', r'in Sachen$', r'Ausgangslage$'],
            Section.CONSIDERATIONS: [r'Erwägungen$'],
            Section.RULINGS: [r'Dispositiv$', r'verfügt die WEKO', r'^[1-9] Ergebnis$', r'^[A-Z] Schlussfolgerungen$'],
            Section.FOOTER: [r'^Rechtsmittelbelehrung:?$']
        },
        Language.FR: {
            Section.HEADER: [],
            Section.FACTS: [r'Etat de fait$', r'in Sachen$'],
            Section.CONSIDERATIONS: [r'Considérants$', r'CONSIDERANTS$'],
            Section.RULINGS: [r'Dispositif$', r'DISPOSITIF$'],
            Section.FOOTER: [r'Voie de droit:?$']
        },
        Language.EN: {
            Section.HEADER: [],
            Section.FACTS: [],
            Section.CONSIDERATIONS: [],
            Section.RULINGS: [],
            Section.FOOTER: []
        },
        
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace)


def VD_Omni(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.FR: {
            Section.FACTS: [r'Vu les faits suivants', r'Vu les faits suivants:', r'constate en fait :',
                            r'Vu les faits suivants :', r'vu les faits suivants :', r'En fait :'],
            Section.CONSIDERATIONS: [
                r'Considérant en droit:', r'Considérant en droit', r'Considérant en droit :',
                r'et considère en droit :', r'^considérant$', r'Considère en droit :', r'Considérant', r'En droit :',
                r'constate ce qui suit en fait et en droit :'],
            Section.RULINGS: [r'du Tribunal cantonal arrête:', r'du Tribunal cantonal arrête:',
                              r'Par ces motifs arrête:', r'Par ces motifs', r'Par ces motifs,'],
            Section.FOOTER: [r'Le président: La greffière:', r'Le président :',
                             r'Le président:', r'Le président: Le greffier:', r'La présidente: La greffière:',
                             r'La présidente:', r'Au nom du Tribunal administratif :', r'La présidente: Le greffier:',
                             r'Le président : Le greffier :']
        },
        Language.EN: {
            Section.FACTS: [],
            Section.CONSIDERATIONS: [],
            Section.RULINGS: [],
            Section.FOOTER: []
        }
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)

    divs = decision.find_all(
        "div", class_=['WordSection1', 'Section1', 'WordSection2'])
    paragraphs = get_paragraphs(divs)

    return associate_sections(paragraphs, section_markers, namespace)


def TI_Gerichte(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.IT: {
            Section.FACTS: [r'ritenuto', r'in fatto ed in diritto', r'ritenuto, in fatto', r'Fatti', r'in fatto:',
                            r'in fatto', r'ritenuto in fatto', r'considerato in fatto e in diritto', r'in fatto'],
            Section.CONSIDERATIONS: [
                r'Diritto', r'in diritto', r'^Considerato$', r'^Considerando$', r'in diritto:',
                r'Considerato, in diritto', r'Considérant', r'En droit :',
                r'constate ce qui suit en fait et en droit :', r'considerato, in diritto', r'^considerando$',
                r'^In diritto:$'],
            Section.RULINGS: [r'Per questi motivi,:', r'Per questi motivi', r'dichiara e pronuncia:', r'pronuncia$',
                              r'pronuncia:$', r'Per i quali motivi,', r'Per i quali motivi', r'^decide:$',
                              r'per questi motivi,', r'pronuncia: 1\.'],
            Section.FOOTER: [r'Per il Tribunale cantonale delle assicurazioni', r'Il presidente La segretaria', r'Per il Tribunale cantonale amministrativo', r'Per la seconda Camera civile del Tribunale d’appello', r'Il presidente La vicecancelliera']
        },
        Language.EN: {
            Section.FACTS: [],
            Section.CONSIDERATIONS: [],
            Section.RULINGS: [],
            Section.FOOTER: []
        },
        Language.FR: {
            Section.FACTS: [],
            Section.CONSIDERATIONS: [],
            Section.RULINGS: [],
            Section.FOOTER: []
        },
        Language.DE: {
            Section.FACTS: [],
            Section.CONSIDERATIONS: [],
            Section.RULINGS: [],
            Section.FOOTER: []
        }
        
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)

    paragraphs = get_paragraphs_unified(decision)

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
            Section.CONSIDERATIONS: [
                r'Aus den Erwägungen:', r'Aus den Erwägungen des Bundesgerichts:', r'Erwägungen:']
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
        paragraphs_by_section[Section.HEADER].append(
            decision[:sorted_section_pos[0]])

    # Assign the corresponding part of the decision to its section
    for i, match_start in enumerate(sorted_section_pos):
        actual_section = sections_found[match_start]
        from_ = match_start
        if i >= len(sorted_section_pos) - 1:
            # This is the last section, till end of decision
            to_ = len(decision)
        else:
            to_ = sorted_section_pos[i + 1]
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
            Section.HEADER: [
                r'(Entscheid|Urteil|Zwischenentscheid|Beschluss|Abschreibungsentscheid|Abschreibungsverfügung) vom \d*\. (Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember) \d*',
                r'\d*\.\/\d*\.\s(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s\d*'],
            Section.FACTS: [r'Sachverhalt:', r'Prozessgeschichte:', r'Nach Einsicht:'],
            Section.CONSIDERATIONS: [r'Erwägungen:'],
            Section.RULINGS: [r'Rechtsspruch:',
                              r'(Demgemäss|Demnach) (beschliesst|erkennt|verfügt) (die|das) (Obergericht|Verfahrensleitung|Verwaltungsgericht|Prozessleitung)(:)*'],
            Section.FOOTER: [
                r'Stans\,\s\d*\.\s(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s\d*']
        }
    }

    if namespace['language'] not in all_section_markers:
        message = f"This function is only implemented for the languages {list(all_section_markers.keys())} so far."
        raise ValueError(message)

    section_markers = prepare_section_markers(all_section_markers, namespace)

    paragraphs = get_pdf_paragraphs(decision)
    return associate_sections(paragraphs, section_markers, namespace)



def BE_Verwaltungsgericht(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[
        Dict[Section, List[str]]]:
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

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)

    paragraphs = get_pdf_paragraphs(decision)
    return associate_sections(paragraphs, section_markers, namespace)


def GR_Gerichte(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [r'Sachverhalt:?$', r'betreffend\s*\w*$' r'hat sich ergeben:?', r'in Sachen$'],
            Section.CONSIDERATIONS: [r'Erwägungen:?$', r'zieht in Erwägung:?$', r'In Erwägung,'],
            Section.RULINGS: [r'^Demnach erkennt', r'wird erkannt:?$', r'^erkannt:?$', r'^Demnach verfügt', r'^verfügt:$', r'^erkannt :$', r'^verfügt :$', r'wird verfügt:$' ],
            Section.FOOTER: [
                r'Für den Kantonsgerichtsausschuss von Graubünden']
        },
        Language.IT: {
            Section.HEADER: [r'TRIBUNALE AMMINISTRATIVO DEL CANTONE DEI GRIGIONI', r'Tribunale cantonale dei Grigioni',
                             r'Dretgira chantunala dal Grischun'],
            Section.FACTS: [r'concernente\s*\w*\s*\w*'],
            Section.CONSIDERATIONS: [r'\s*Considerando\s*in\s*diritto\s*:\s*',
                                     r'(in )*constatazione e in considerazione,',
                                     r'La (Presidenza|Commissione) del Tribunale cantonale considera :',
                                     r'Considerandi',
                                     r'La Camera (di gravame|civile) considera :', r'In considerazione:',
                                     r'visto e considerato:',
                                     r'Considerato in fatto e in diritto:', r'^((La|Il)\s(\w+\s)*en consideraziun:)$'],
            Section.RULINGS: [
                r'^(((L|l)a (Prima|Seconda) )*Camera (penale|civile) (pronuncia|giudica|decreta|decide|ordina|considera)\s*:)',
                r'Decisione \─ Dispositivo', r'Per questi motivi il Tribunale giudica:', r'Il Tribunale decide:',
                r'La (Presidenza|Commissione) del Tribunale cantonale (ordina|giudica:)',
                r'La Camera di gravame (considera|decide) :', r'Per questi motivi si decreta:',
                r'(La )*Camera civile giudica:', r'decide:', r'(la Presidenza )ordina\s*(:)*', r'(Si )*giudica',
                r'La Camera delle esecuzioni e dei fallimenti decide:', r'(i|I)l Giudice unico decide:',
                r'decreta', r'^((La|Il)\s(\w+\s)*decida damai:)$', r'^(è giudicato:)$'],
            Section.FOOTER: [
                r'Per la Presidenza del Tribunale cantonale dei Grigioni']
        }
    }

    valid_namespace(namespace, all_section_markers)
    section_markers = prepare_section_markers(all_section_markers, namespace)

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
            Section.CONSIDERATIONS: [r'^Begründung:\s*$', r'Erwägung(en)?:?\s*$', r'^Entscheidungsgründe$',
                                     r'[iI]n Erwägung[:,]?\s*$'],
            Section.RULINGS: [r'Demgemäss erkennt d[\w]{2}', r'erkennt d[\w]{2} [A-Z]\w+:',
                              r'Appellationsgericht (\w+ )?(\(\w+\) )?erkennt', r'^und erkennt:$', r'erkennt:\s*$'],
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

def VS_Gerichte(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.FACTS: [r'^[I,i]n Sachen', r'^Sachverhalt:?$', r'Sachverhalt \(gekürzt\)', r'Gekürzter Sachverhalt', r'Sachverhalt und Verfahren', r'SACHVERHALT', r'^Procédure$', r'^Verfahren$'],
            Section.CONSIDERATIONS: [r'^Erwägungen:?$', r'Aus den Erwägungen', 'Sachverhalt und Erwägungen', r'stellt fest und zieht in Erwägung', r'ERWÄGUNGEN'],
            Section.RULINGS: [r'erkennt[:]?$', r'Demnach erkennt das Kantonsgericht:', r'Demnach wird erkannt',
                              r'Demnach wird erkannt:', r'Demnach erkennt das Kantonsgericht', r'Das Kantonsgericht beschliesst', r'Das Kantonsgericht verfügt', r'DEMNACH WIRD ERKANNT:'],
            Section.FOOTER: [r'no footer available']
        },
        Language.FR: {
            Section.FACTS: [r'Faits$', r'Faits \(résumé\)', r'FAITS ET PROCEDURE', r'Faits et procédure', r'Statuant en faits', r'^Vu$'],
            Section.CONSIDERATIONS: [r'Considérant en droit:?', r'Considérants \(extraits\)', r'^[C,c]onsidérant$', r'DROIT', r'^Droit$'],
            Section.RULINGS: [r'[P,p]ar ces motifs,', r'^[P,p]rononce:?$', r'PRONONCE'],
            Section.FOOTER: [r'no footer available']
        }
    }
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)

    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace)


def SZ_Gerichte(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    # As soon as one of the strings in the list (regexes) is encountered we switch to the corresponding section (key)
    # (?:C|c) is much faster for case insensitivity than [Cc] or (?i)c
    all_section_markers = {
        Language.DE: {
            Section.FACTS: [r'no fact section'],
            Section.CONSIDERATIONS: [r'nachdem sich ergeben', r'nachdem sich ergeben und in Erwägung:', 'in Erwägung'],
            Section.RULINGS: [r'^erkennt[:]?$', r'^beschlossen[:]?$', r'^verfügt[:]?$', r'^erkannt[:]?$',
                              r'erkannt und beschlossen[:]?$', r'beschlossen und erkannt[:]?$'],
            Section.FOOTER: [r'^Namens', r'^Versand']
        }
    }
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)

    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace)


def SO_Omni(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [
                r'^((Beschluss|Urteil|Entscheid)\svom\s\d*\.*\s(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s\d*)',
                r'^((SOG|KSGE) \d* Nr\. \d*)$'],
            Section.FACTS: [r'^(Sachverhalt\s*(gekürzt)*:*)$', r'^(In Sachen)'],
            Section.CONSIDERATIONS: [r'^((Aus den )*Erwägungen:*)$',
                                     r'^(zieht\s\w+\s*(.+)\s*Erwägung(en)*(:)*(, dass)*(:)*)',
                                     r'^((Die|Der|Das)\s(\w+\s)*zieht in Erwägung:)$'],
            Section.RULINGS: [r'^(Demnach wird (erkannt|beschlossen|verfügt):)$', r'^(erkannt:)$',
                              r'^((beschlossen|festgestellt) und erkannt:)',
                              r'^(Demnach wird\s\w+\s*(.+)\s*(beschlossen|erkannt|verfügt):)'],
            Section.FOOTER: [r'^(Rechtsmittel(\sbelehrung)*(:)*)']
        }
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)

    divs = decision.find_all(
        "div", class_=['WordSection1'])
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
            Section.TOPIC: [r'Gegenstand', r'betreffend', r"Betreff", r"wegen"],
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
            Section.TOPIC: [r'Objet'],
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
            Section.TOPIC: [r'Oggetto'],
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


def CH_BSTG(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [
                r'^((Verfügung|Beschluss|Urteil|Entscheid|Präsidialverfügung|Präsidialentscheid) vom \d*\.* (Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s\d*)',
                r'^(Präsidialentscheid vom )$'],
            Section.FACTS: [r'^(Sachverhalt(:)*)$', r'Prozessgeschichte(:)*',
                            r'(Die|Der)\s\w+\s*(.+)\s*hält fest, dass\s*(:*)'],
            Section.CONSIDERATIONS: [r'^(Nach Einsicht in)$', r'^([iI]n\sErwägung(:)*)', r'^(Erwägungen:*)$',
                                     r'(Die|Der|Das|Aus)(\s([\w\.])*)*\s[eE]rwäg\w*(\,\sdass)*\s*(:)*\s*'],
            Section.RULINGS: [r'^(und (verfügt|erkennt|beschliesst)(:)*\s*)$',
                              r'^(p*(Die|Der|Das|und)(\s(\w|\.)*)*\s(verfügt|erkennt|beschliesst)(:)*)$',
                              r'^(Demnach (erkennt|verfügt|beschliesst)\s\w+\s*(.+)\s\w*(:)*)',
                              r'^(beschliesst die Strafkammer:)$'],
            Section.FOOTER: [r'^(Rechtsmittelbelehrung)',
                             r'^(Hinweis:*( auf Art\. 78 BGG| auf das Bundesgerichtsgesetz \(BGG, SR 173\.110\)| auf die Rechtsmittelordnung)*)$',
                             r'^(Beschwerde an die Beschwerdekammer des Bundesstrafgerichts)',
                             r'^(Nach Eintritt der Rechtskraft mitzuteilen an:*)', r'^(Zustellung an\s*)$',
                             r'Gegen Entscheide der Strafkammer des Bundesstrafgerichts']
        },

        Language.FR: {
            Section.HEADER: [
                r'^((Arrêt|Ordonnance|Décision|Jugement) du \d*\W*\s(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s\d*)$'],
            Section.FACTS: [r'^((F|f)(AITS|aits)(:)*)',
                            r'(La|Le)*(\s\w*)*\s*(\,\s)*(V|v)u\s*(:)*(que)*(:)*(\sle dossier de la cause)*'],
            Section.CONSIDERATIONS: [r'(et|Et)*\s*(C|c)onsidérant\s*(que)*:\s*', r'La Cour d’appel considère(\s)*:',
                                     r'DROIT', r'(La|Le|Considérant)(\s\w+\s*(.+)\s*)(considère|et) en droit:'],
            Section.RULINGS: [r'Ordonne:', r'(La|Le)\s\w+\s*(.+)\s(prononce|décide)\s*(:)', r'^(pronnonce\s*(:))',
                              r'Par ces motifs\,(\s\w*)*\s(prononce|décide|ordonne)\s*:\s*'],
            Section.FOOTER: [r'Indication(s)* des voies de (recours|droit|plainte)', r'Voies de droit',
                             r'^(Distribution\s*(\(\s*acte judiciaire\)):*\s*)',
                             r'Appel à la Cour d’appel du Tribunal pénal fédéral',
                             r'^(Une expédition complète de la décision est adressée à)',
                             r'^(Distribution\s*(\(\s*recommandé\)):*\s*)', r'^(Notification des voies de recours)']
        },

        Language.IT: {
            Section.HEADER: [
                r'^((Sentenza|Decisione(\ssupercautelare)*|Ordinanza|Decreto)\s*del(l)*\W*\d*\W*\s*(gennaio|febbraio|marzo|aprile|maggio|luglio|agosto|settembre|ottobre|novembre|dicembre|giugno)\s*\d*)$'],
            Section.FACTS: [r'^([Ff]att(i|o)\s*:)$', r'Visti:', r'^(\w+\s*(.+)\s*penali, vist(o|i)\s*(:)*)',
                            r'(Ritenuto )*in fatto( e(d)* in diritto):'],
            Section.CONSIDERATIONS: [r'^((e\s)*[Cc]onsiderato:?\s*)$', '^([Dd]iritto(:)*\s*)$',
                                     r'^(La Corte considera in fatto e in diritto:)',
                                     r'La Corte(\sd(\'|\’)appello)* considera in diritto:', r'^(In diritto:)$',
                                     r'Estratto dei considerandi:'],
            Section.RULINGS: [r'La Corte (decreta|pronuncia|ordina):',
                              r'^(Per questi motivi(\,)*(\s\w*)*\s(decreta|ordina|pronuncia):)$',
                              r'^((Per questi motivi, )*[Ll]a I(I)*(\.)* Corte dei reclami penali pronuncia:\s*)$',
                              r'Il Giudice unico pronuncia:', r'^(Decreta:)$', r'^(Il Presidente decreta:)'],
            Section.FOOTER: [r'(Informazione\ssui\s)*[Rr]imedi\sgiuridici', r'^(Intimazione a:)',
                             r'^(Il testo integrale della sentenza viene notificato a:)', r'^(Comunicazione a(:)*\s*)$',
                             r'Reclamo alla Corte dei reclami penali del Tribunale penale federale',
                             r'^(Comunicazione \(atto giudiziale\) a:)']
        }
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)

    paragraphs = get_pdf_paragraphs(decision)
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
                if paragraph is not None:
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
        element = element.replace('  ', ' ')
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
        # Should exit exec here
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


def FR_Gerichte(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [r'Sachverhalt'],
            Section.CONSIDERATIONS: [r'Erwägungen', r' zieht in Erwägung,'],
            Section.RULINGS: [r'erkennt:$',r'erkennt der Hof:', r'entscheidet:$'],
            Section.FOOTER: [r'Gegen diesen Entscheid kann innerhalb', r'Dieses Urteil kann innert', r'Gegen die Festsetzung der Höhe der Verfahrenskosten', r'Gegen diesen Entscheid kann innert 30 Tagen', r'innert 30 Tagen']
        },
        Language.FR: {
            Section.HEADER: [],
            Section.FACTS: [r'considérant en fait', r'^attendu$'],
            Section.CONSIDERATIONS: [r'considérant en fait et en droit', r'en droit$', r'^considérant$'],
            Section.RULINGS: [r'la Cour arrête', r'prononce:$', r'la Chambre arrête', r'arrête:?$'],
            Section.FOOTER: [r'Cet arrêt peut faire', r'Cette décision peut', r'Siégeant', r'Voie de recours', r'Le recours doit être', r'dans un délai de 30 jours', r'dans les 30 jours']
        },
    }
 
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace)

def OW_Gerichte(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.RULINGS: [],
            Section.FACTS: [r'Sachverhalt'],
            Section.CONSIDERATIONS: [r'den Erwägungen', r'In Erwägung:', r'Erwägungen:', r'vorstehenden Erwägungen'],
            Section.FOOTER: [r'^Rechtsmittelbelehrung']
        },
    }
 

    custom_order = [Section.RULINGS, Section.FACTS, Section.CONSIDERATIONS]
    
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace, custom_order)

def CH_EDOEB(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [r'in Sachen', r'Sachverhalt', r'und Öffentlichkeitsbeauftragte stellt fest'],
            Section.CONSIDERATIONS: [r'Erwägungen',  r'Erwägung'],
            Section.RULINGS: [r'[A,a]ufgrund dieser Erwägungen empfiehlt'],
            Section.FOOTER: [r'Rechtsmittelbelehrung']
        },
        Language.FR: {
            Section.HEADER: [],
            Section.FACTS: [r'données et à la transparence constate'],
            Section.CONSIDERATIONS: [r'Considérants formels :', r'considère ce qui suit :'],
            Section.RULINGS: [r'recommande ce qui suit'],
            Section.FOOTER: [r'Rechtsmittelbelehrung']
        },
        Language.IT: {
            Section.HEADER: [],
            Section.FACTS: [r'federale della protezione dei dati e della trasparenza accerta quanto segue'],
            Section.CONSIDERATIONS: [r'protezione dei dati e della trasparenza considera quanto'],
            Section.RULINGS: [r'trasparenza formula le seguenti raccomandazioni'],
            Section.FOOTER: [r'Rechtsmittelbelehrung']
        },
        Language.EN: {
            Section.HEADER: [],
            Section.FACTS: [],
            Section.CONSIDERATIONS: [],
            Section.RULINGS: [],
            Section.FOOTER: []
        },
    }
 
    
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace)


def SH_OG(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [r'Sachverhalt'],
            Section.CONSIDERATIONS: [r'Aus den Erwägungen'],
            Section.RULINGS: [r'no rulings section'],
            Section.FOOTER: [r'Rechtsmittelbelehrung']
        },
    }
    
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace)

def VD_FindInfo(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.FR: {
            Section.HEADER: [],
            Section.FACTS: [r'En fait', r'E n f a i t', r'EN FAIT', r'Vu l\'enquête'],
            Section.CONSIDERATIONS: [r'E n d r o i t', r'En droit', r'En fait et en droit'],
            Section.RULINGS: [r'ces motifs,$'],
            Section.FOOTER: [r'l\'envoi de photocopies.',r'Le greffier', r'L\'arrêt qui précède, dont la rédaction a été approuvée à huis', r'La greffière', r'L’arrêt est exécutoire']
        },
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [r'Sachverhalt'],
            Section.CONSIDERATIONS: [r'Aus den Erwägungen'],
            Section.RULINGS: [r'no rulings section'],
            Section.FOOTER: [r'Rechtsmittelbelehrung']
        },
    }
    
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace)

def LU_Gerichte(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [r'no fact section'],
            Section.RULINGS: [r'^Entscheid:$'],
            Section.CONSIDERATIONS: [r'Aus den Erwägungen'],
            Section.FOOTER: [r'no footer section']
        },
    }
    
    custom_order = [Section.HEADER, Section.FACTS, Section.RULINGS, Section.CONSIDERATIONS, Section.FOOTER]
    
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace, custom_order)


def JU_Gerichte(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.FR: {
            Section.HEADER: [],
            Section.FACTS: [r'EN FAIT', r'En fait', r'Vu l[e,a]'],
            Section.RULINGS: [r'PAR CES MOTIFS'],
            Section.CONSIDERATIONS: [r'En droit', r'EN DROIT'],
            Section.FOOTER: [r'AU NOM DE LA COUR ADMINISTRATIVE', r'Communication concernant les moyens de recours']
        },
    }
    
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace)

def CH_BVGer(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [r'Sachverhalt', r'Das Bundesverwaltungsgericht stellt fest,'],
            Section.CONSIDERATIONS: [r'zieht in Erwägung', r'stellt fest und erwägt'],
            Section.RULINGS: [r'Demnach erkennt das Bundesverwaltungsgericht'],
            Section.FOOTER: [r'Der vorsitzende Richter: Der Gerichtsschreiber:', r'Der vorsitzende Richter: Die Gerichtsschreiberin:', r'Dieses Urteil geht an:', r'Gegen diesen Entscheid kann innert 30 Tagen nach', r'^Rechtsmittelbelehrung:$']
        },
        Language.FR: {
            Section.HEADER: [],
            Section.FACTS: [r'Sachverhalt', r'Das Bundesverwaltungsgericht stellt fest,'],
            Section.CONSIDERATIONS: [r'et considérant', r'Droit :$'],
            Section.RULINGS: [r'Par ces motifs, le Tribunal administratif fédéral prononce', r'Tribunal administratif fédéral prononce'],
            Section.FOOTER: [r'Indication des voies de droit :', r'Le juge unique : Le greffier :', r'Voies de droit:', r'La présidente du collège : Le greffier :', r'La juge unique : Le greffier :']
        },
        Language.IT: {
            Section.HEADER: [],
            Section.FACTS: [r'Fatti:', r'Ritenuto in fatto', r'Visto:?$'],
            Section.RULINGS: [r'Per questi motivi, il Tribunale amministrativo federale', r'Per questi motivi, il Tribunale amministrativo federale pronuncia'],
            Section.CONSIDERATIONS: [r'Ritenuto in fatto e considerato in diritto:', r'Diritto:', r'e considerato', r'Considerando in diritto'],
            Section.FOOTER: [r'Il presidente del collegio:', r'La presidente del collegio:', r'Rimedi di diritto', r'Data di spedizione:']
        },
        Language.EN: {
            Section.HEADER: [],
            Section.FACTS: [],
            Section.CONSIDERATIONS: [],
            Section.RULINGS: [],
            Section.FOOTER: []
        },
    }
    
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace)


def GR_Gerichte(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [r'Sachverhalt', r'hat sich ergeben:'],
            Section.CONSIDERATIONS: [r'zieht in Erwägung', r'Aus den Erwägungen:'],
            Section.RULINGS: [r'Demnach erkennt', r'Demnach wird erkannt', r'^erkennt:$', r'wird erkannt:$', r'^verfügt:$', r'Demnach wird verfügt:'],
            Section.FOOTER: [r'\[Rechtsmittelbelehrung\]', r'(Rechtsmittelbelehrung)']
        },
        Language.IT: {
            Section.HEADER: [],
            Section.FACTS: [r'Fattispecie', r'Ritenuto in fatto', r'Fattispecie'],
            Section.CONSIDERATIONS: [r'Considerando in diritto:', r'Droit :$', r'Considerandi'],
            Section.RULINGS: [r'Per questi motivi', r'Tribunale decide:'],
            Section.FOOTER: [r'Indication des voies de droit :', r'Le juge unique : Le greffier :', r'Voies de droit:', r'La présidente du collège : Le greffier :', r'La juge unique : Le greffier :']
        },  
    }
    
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)

    return associate_sections(paragraphs, section_markers, namespace)

def NE_Omni(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.FR: {
            Section.HEADER: [],
            Section.FACTS: [r'Résumé'],
            Section.CONSIDERATIONS: [r'C O N S I D E R A N T', r'C O N S I D E RA N T', r'CO N S I D E R A N T', r'en droit', r'e n  d r o i t'],
            Section.RULINGS: [r'Par ces motifs',r'Par cesmotifs'],
            Section.FOOTER: [r'Le greffier',r'AU NOM DU TRIBUNAL ADMINISTRATIF', r'^Neuchâtel, le ']
        }
    }
 
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)
    

    return associate_sections(paragraphs, section_markers, namespace)


def SG_Publikationen(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [r'Sachverhalt', r'in Sachen', r'Aus dem Sachverhalt:', r'Zum Sachverhalt:', r'hat das Verwaltungsgericht festgestellt:', r'Das Verwaltungsgericht stellt fest:'],
            Section.CONSIDERATIONS: [r'Erwägungen', r'in Erwägung', r'Darüber zieht das Verwaltungsgericht in Erwägung:', r'Der Abteilungspräsident erwägt:'],
            Section.RULINGS: [r'zu Recht erkannt:',r'Demnach erkennt das Verwaltungsgericht zu Recht:', r'Demnach erkennt das Verwaltungsgericht auf dem Zirkulationsweg zu Recht:', r'Demnach hat das Verwaltungsgericht zu Recht erkannt:', r'Der Abteilungspräsident verfügt:', r'Der Präsident verfügt:'
                              , r'zu Recht:$', r'verfügt:$', r'Entscheid:$', r'entschieden:$', r'^Entscheid$' ],
            Section.FOOTER: [r'Rechtsmittelbelehrung']
        }
    }
 
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)
    
    return associate_sections(paragraphs, section_markers, namespace)

def SG_Gerichte(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [r'Sachverhalt:?$', r'in Sachen$', r'Das Verwaltungsgericht stellt fest:', r'hat das Verwaltungsgericht festgestellt:'],
            Section.CONSIDERATIONS: [r'^Erwägungen:?$', r'^Erwägung$', r'Darüber wird in Erwägung gezogen:', r'Darüber zieht das Verwaltungsgericht in Erwägung:', r'Aus den Erwägungen:', r'hat das Versicherungsgericht in Erwägung gezogen:', r'Der Abteilungspräsident erwägt:', r'in Erwägung gezogen:'],
            Section.RULINGS: [r'^Entscheid:?$', r'^entschieden:?$', r'^erkannt:?$', r'zu Recht erkannt:', r'zu Recht:$', r'zu Recht erkannt:$', r'^beschlossen$', r'festgestellt und erkannt:?$', r'verfügt:$', r'beschlossen und erkannt:?$', r'beschlossen:$', r'Demgemäss hat das Versicherungsgericht entschieden:' ],
            Section.FOOTER: [r'Rechtsmittelbelehrung']
        }
    }
 
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)
    
    return associate_sections(paragraphs, section_markers, namespace)




def associate_sections(paragraphs: List[str], section_markers, namespace: dict,
                       sections: List[Section] = list(Section), current_section = Section.HEADER):
    """
    Associate sections to paragraphs
    :param paragraphs:      list of paragraphs
    :param section_markers: dict of section markers
    :param namespace:       dict of namespace
    :param sections:        if some sections are not present in the court, pass a list with the missing section excluded
    """
    paragraphs_by_section = {section: [] for section in sections}
    for paragraph in paragraphs:
        # update the current section if a marker of a different section matched
        current_section = update_section(
            current_section, paragraph, section_markers, sections)
        # add paragraph to the list of paragraphs
        paragraphs_by_section[current_section].append(paragraph)

    if current_section != Section.FOOTER:
        exceptions = ['ZH_Steuerrekurs', 'GL_Omni']  # Has no footer
        if not namespace['court'] in exceptions:
            # change the message depending on whether there's a url
            if namespace.get('html_url'):
                message = f"({namespace['id']}): We got stuck at section {current_section}. Please check! " \
                          f"Here you have the url to the decision: {namespace['html_url']}"
            elif 'pdf_url' in namespace and namespace['pdf_url']:
                message = f"({namespace['id']}): We got stuck at section {current_section}. Please check! " \
                          f"Here is the url to the decision: {namespace['pdf_url']}"
            else:
                message = f"({namespace['id']}): We got stuck at section {current_section}. Please check! "
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
    paragraph = unicodedata.normalize(
        'NFC', paragraph)  # if we don't do this, we get weird matching behaviour
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



def CH_BGE(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.FACTS: [r'Sachverhalt'],
            Section.CONSIDERATIONS: [r'Erwägungen', r'in Erwägung:', r'Erwägung:'],
            Section.RULINGS: [r'Dispositiv', r'Demnach erkennt', r'Demnach beschliesst', r'wird beschlossen:', r'wird verfügt:', r'erkannt:'],
            Section.FOOTER: [r'^Rechtsmittelbelehrung']
        },
        Language.FR: {
            Section.FACTS: [r'Sachverhalt', r'Résumé des faits'],
            Section.CONSIDERATIONS: [r'Erwägungen', r'Extrait des considérants', r'Considérant en droit', r'Extraits des considérants', r'motifs suivants'],
            Section.RULINGS: [r'Dispositiv', r'Par ces motifs', r'motifs suivants'],
            Section.FOOTER: [r'^Rechtsmittelbelehrung']
        },
        Language.IT: {
            Section.FACTS: [r'Sachverhalt'],
            Section.CONSIDERATIONS: [r'Erwägungen'],
            Section.RULINGS: [r'Dispositiv'],
            Section.FOOTER: [r'^Rechtsmittelbelehrung']},
        Language.EN: {
            Section.HEADER: [],
            Section.FACTS: [],
            Section.CONSIDERATIONS: [],
            Section.RULINGS: [],
            Section.FOOTER: []
        },
    }
 
    
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)
    
    return associate_sections(paragraphs, section_markers, namespace)

def AI_Aktuell(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [r'no facts section'],
            Section.CONSIDERATIONS: [r'^Erwägungen:?$'],
            Section.RULINGS: [r'no ruling section'],
            Section.FOOTER: [r'^Rechtsmittelbelehrung']
        }
    }
 
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)
    
    return associate_sections(paragraphs, section_markers, namespace)

def AI_Bericht(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [r'no facts section'],
            Section.CONSIDERATIONS: [r'^Erwägungen:?$', r'Aus den Erwägungen der Standeskommission:'],
            Section.RULINGS: [r'no ruling section'],
            Section.FOOTER: [r'^Rechtsmittelbelehrung']
        }
    }
 
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)
    
    return associate_sections(paragraphs, section_markers, namespace)

def AR_Gerichte(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.HEADER: [],
            Section.FACTS: [r'^Sachverhalt:?$'],
            Section.CONSIDERATIONS: [r'^Erwägungen:?$', r'Aus den Erwägungen:'],
            Section.RULINGS: [r'erkennt das Obergericht:', r'erkennt:?$', r'beschliesst:?$', r'beschliesst das Obergericht:',],
            Section.FOOTER: [r'^Rechtsmittelbelehrung', r'4. Rechtsmittel:', r'Der Obergerichtsvizepräsident:', r'Der Obergerichtspräsident:']
        }
    }
 
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)
    
    return associate_sections(paragraphs, section_markers, namespace)


def find_ruling(namespace, sections):
    section_markers = prepare_section_markers(all_judgment_markers, namespace)
    latestSection = Section.HEADER
    index = -1
    for section in Section:
        if section is not Section.FOOTER:
            if len(sections[section]) > 0:
                latestSection = section
    for idx, paragraph in enumerate(reversed(sections[latestSection])):
        for key in section_markers:
            marker = section_markers[key]
            if re.search(marker, paragraph):
                index = idx
    sections[Section.RULINGS] = sections[latestSection][index:]
    del sections[latestSection][index:]
    return sections
            
    


def ZG_Verwaltungsgericht(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[
        Dict[Section, List[str]]]:
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
            Section.RULINGS: [r'Demnach erkennt', r'Folgendes verfügt', r'(Der|Die|Das) \w+ verfügt:',
                              r'Demnach wird verfügt:', r'Demnach wird erkannt'],
            Section.FOOTER: [
                r'^[\s]*Zug,( den| vom)?\s\d?\d\.?\s?(?:Jan(?:uar)?|Feb(?:ruar)?|Mär(?:z)?|Apr(?:il)?|Mai|Jun(?:i)?|Jul(?:i)?|Aug(?:ust)?|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?|Dez(?:ember)?)\s\d{4}']
        }
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)

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
            Section.RULINGS: [r'(Zusammengefasst|Zusammenfassend) (ist|sind)',
                              r'(Zusammengefasst|Zusammenfassend) ergibt sich', r'Der Rekurs ist nach',
                              r'Gesamthaft ist der Rekurs',
                              r'Dies führt zur (Aufhebung|Abweisung|Gutheissung|teilweisen)'],
            # there are generally no footers
            Section.FOOTER: [r'Im Namen des Baurekursgerichts']
        },
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)

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
            Section.CONSIDERATIONS: [r'(?:A|a)us den Erwägungen ', r'Erwägungen:', r'^[\s]*Erwägungen[\s]*$',
                                     r'Das (Einzelgericht|Gericht) erwägt',
                                     r'Das (Einzelgericht|Gericht) zieht in (Erwägung|Betracht)',
                                     r'hat in Erwägung gezogen:'],
            Section.RULINGS: [r'^[\s]*Es wird (erkannt|beschlossen):', r'^[\s]*wird beschlossen:[\s]*$',
                              r'Das (Einzelgericht|Gericht) (erkennt|beschliesst):',
                              r'(Sodann|Demnach|Demgemäss) beschliesst das Gericht:'],
            Section.FOOTER: [
                r'^[\s]*Zürich,( den| vom)?\s\d?\d\.?\s?(?:Jan(?:uar)?|Feb(?:ruar)?|Mär(?:z)?|Apr(?:il)?|Mai|Jun(?:i)?|Jul(?:i)?|Aug(?:ust)?|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?|Dez(?:ember)?)\s\d{4}([\s]*$)',
                r'OBERGERICHT DES KANTONS ZÜRICH']
        }
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)

    paragraphs = get_pdf_paragraphs(decision)
    return associate_sections(paragraphs, section_markers, namespace)


def ZH_Sozialversicherungsgericht(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[
        Dict[Section, List[str]]]:
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
            Section.FACTS: [r'Sachverhalt:'],
            Section.CONSIDERATIONS: [r'in Erwägung', r'Erwägungen:'],
            Section.RULINGS: [r'Das Gericht (erkennt|beschliesst|verfügt):',
                              r'(Der|Die) Einzelrichter(in)? (erkennt|beschliesst|verfügt):',
                              r'(beschliesst|erkennt) das Gericht:', r'und erkennt sodann:',
                              r'(Der|Die) Referent(in)? (erkennt|beschliesst|verfügt):'],
            Section.FOOTER: [r'Gegen diesen Entscheid kann']
        }
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)
    
    return associate_sections(paragraphs, section_markers, namespace)


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
            Section.RULINGS: [r'Demgemäss (erkennt|beschliesst|verfügt)', r'beschliesst die Rekurskommission',
                              r'verfügt der Einzelrichter', r'verfügt die Einzelrichterin'],
            # there is generally no footer
            Section.FOOTER: [
                r'Im Namen des Steuerrekursgerichts']
        }
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)

    paragraphs = get_pdf_paragraphs(decision)
    return associate_sections(paragraphs, section_markers, namespace)


def ZH_Verwaltungsgericht(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[
        Dict[Section, List[str]]]:
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
            Section.FACTS: [r'hat sich ergeben:', r'^\s*I\.\s+A\.\s*', r'^\s*I\.\s+(&nbsp;)?$', r'^\s*I\.\s[A-Z]+',
                            r'nach Einsichtnahme in', r'Sachverhalt[:]?[\s]*$'],
            Section.CONSIDERATIONS: [r'erwägt:', r'zieht in (Erwägung|Betracht)',
                                     r'zieht (der Einzelrichter|die Einzelrichterin) in Erwägung', r'in Erwägung, dass',
                                     r'(?:A|a)us den Erwägungen', r'hat erwogen:'],
            Section.RULINGS: [
                r'(Demgemäss|Demnach|Dementsprechend|Demmäss) (erkennt|erkannt|beschliesst|entscheidet|verfügt)',
                r'Das Verwaltungsgericht entscheidet',
                r'(Die Kammer|Der Einzelrichter|Die Einzelrichterin) (erkennt|entscheidet|beschliesst|hat beschlossen)',
                r'Demgemäss[\s|(&nbsp;)]*die Kammer:', r'Der Abteilungspräsident verfügt:', r'^[\s]*verfügt[:]?[\s]*$',
                r'^[\s]*entschieden:[\s]*$', r'^[\s]*und (entscheidet|erkennt):[\s]*$'],
            # this court generally has no footer
            Section.FOOTER: [r'Rechtsmittelbelehrung']
        },
            Language.FR: {
            # "header" has no markers!
            Section.FACTS: [r'En fait'],
            Section.CONSIDERATIONS: ['En droit'],
            Section.RULINGS: ['Par ces motifs:'],
            # this court generally has no footer
            Section.FOOTER: [r'Voie de recours']
        }
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)

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
        assert (len(divs) <= 2), "Found more than two divs with class WordSection1"
        assert (len(divs) > 0), "Found no div, " + str(namespace['html_url'])

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
                # only clean and append non-empty paragraphs
                if paragraph not in ['', ' ', None]:
                    paragraph = clean_text(paragraph)
                    paragraphs.append(paragraph)
        return paragraphs

    paragraphs = get_paragraphs(decision)
    return associate_sections(paragraphs, section_markers, namespace)


# returns dictionary with section names as keys and lists of paragraphs as values

def BE_BVD(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    all_section_markers = {
        Language.DE: {
            Section.FACTS: [r'Sachverhalt'],
            Section.CONSIDERATIONS: [r'II. Erwägungen'],
            Section.RULINGS: [r'III. Entscheid'],
            Section.FOOTER: [r'IV. Eröffnung']
        },
        Language.FR: {
            Section.FACTS: [r'I. Faits'],
            Section.CONSIDERATIONS: [ r'II. Considérants'],
            Section.RULINGS: [r'III. Décision'],
            Section.FOOTER: [r'IV. Notification']
        },
    }
    
    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)
    
    paragraphs = get_paragraphs_unified(decision)
    
    return associate_sections(paragraphs, section_markers, namespace)


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

    all_section_markers = {
        Language.DE: {
            # "header" has no markers!
            # "facts" are not present either in this court, leave them out
            Section.FACTS: [r'no facts section available'],
            Section.CONSIDERATIONS: [r'^Erwägungen:|^Erwägungen$', r'Auszug aus den Erwägungen', r'Formelles$',
                                     '^Sachverhalt(?: |:)'],
            Section.RULINGS: [r'^Die (?:Aufsichtsbehörde|Kammer) entscheidet:', r'(?:^|\. )Dispositiv',
                              r'^Der Instrkutionsrichter entscheidet:', r'^Strafkammer erkennt:',
                              r'^Die Beschwerdekammer in Strafsachen (?:beschliesst|hat beschlossen):',
                              r'^Das Gericht beschliesst:',
                              r'^Die Verfahrensleitung verfügt:', r'^Der Vizepräsident entscheidet:',
                              r'^Das Handelsgericht entscheidet:', r'^Die \d. Strafkammer beschliesst:', r'Die \d. Strafkammer erkennt:', r'Das Gericht entscheidet:'],
            # "Weiter wird verfügt:" often causes problems with summarys in the considerations section, leave it out
            Section.FOOTER: [r'^Zu eröffnen:', r'\d\. Zu eröffnen:', r'^Schriftlich zu eröffnen:$',
                             r'^Rechtsmittelbelehrung', r'^Hinweis:']  # r'^Weiter wird verfügt:'
        },
        Language.FR: {
            # "header" has no markers!
            # "facts" are not present either in this court, leave them out
                        Section.FACTS: [r'no facts section available'],
            Section.CONSIDERATIONS: [r'^Considérants(?: :|:)?',
                                     r'^Extrait des (?:considérations|considérants)(?: :|:)'],
            Section.RULINGS: [r'^La Chambre de recours pénale décide(?: :|:)', r'^Dispositif'],
            Section.FOOTER: [r'A notifier(?: :|:)', r'Le présent (?:jugement|dispositif) est à notifier(?: :|:)',
                             r'Le présent jugement est à notifier par écrit(?: :|:)']
        }
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)


    paragraphs = get_paragraphs_unified(decision)

    # pass custom sections without facts
    sections = associate_sections(
        paragraphs, section_markers, namespace)

    return sections


def CH_BPatG(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    Remark: This court does not have a facts section, and some don't have a footer.
    """
    all_section_markers = {
        Language.DE: {
            # Section.FACTS: [], # no facts in this court
            Section.FACTS: [],
            Section.CONSIDERATIONS: [
                r'^(?:Das Bundespatentgericht|(?:Der|Das) Präsident|Die Gerichtsleitung|Das Gericht|Der (?:Einzelrichter|Instruktionsrichter))' \
                r' zieht in Erwägung(?:,|:)',
                r'Der Präsident erwägt:', r'Aus(?:|zug aus) den Erwägungen:', r'Sachverhalt:'],
            Section.RULINGS: [
                r'(?:Der Instruktionsrichter|Das Bundespatentgericht|(?:Das|Der) Präsident) (?:erkennt|verfügt|beschliesst)(?:,|:)',
                r'Die Gerichtsleitung beschliesst:', r'Der Einzelrichter erkennt:'],
            Section.FOOTER: [r'Rechtsmittelbelehrung:',
                             r'Dieser Entscheid geht an:']
        },
        Language.FR: {
            # Section.FACTS: [], # no facts in this court
            Section.FACTS: [],
            Section.CONSIDERATIONS: [r'Le Tribunal fédéral des brevets considère(?: :|:|,)',
                                     r'Le [pP]résident considère(?: :|:|,)'],
            Section.RULINGS: [r'Le Tribunal fédéral des brevets décide:', r'Le [pP]résident (décide|reconnaît):'],
            Section.FOOTER: [r'Voies de droit:']
        },
        Language.IT: {
            Section.FACTS: [],
            # Section.FACTS: [], # no facts in this court
            Section.CONSIDERATIONS: [r'Considerando in fatto e in diritto:'],
            Section.RULINGS: [r'Per questi motivi, il giudice unico pronuncia:'],
            Section.FOOTER: [r'Rimedi giuridici:']
        },
            
    }

    valid_namespace(namespace, all_section_markers)

    section_markers = prepare_section_markers(all_section_markers, namespace)

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
            element = element.replace('  ', ' ')
            paragraph = clean_text(element)
            if paragraph not in ['', ' ', None]:
                paragraphs.append(paragraph)
        return paragraphs

    paragraphs = get_paragraphs(decision)

    return associate_sections(paragraphs, section_markers, namespace)
