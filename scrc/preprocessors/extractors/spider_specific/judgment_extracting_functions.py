import unicodedata
from typing import Any, Dict, Optional, List

import re

from scrc.enums.judgment import Judgment
from scrc.enums.language import Language
from scrc.utils.main_utils import clean_text, int_to_roman

"""
This file is used to extract the judgment outcomes from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
Overview of spiders still todo: https://docs.google.com/spreadsheets/d/1FZmeUEW8in4iDxiIgixY4g0_Bbg342w-twqtiIu8eZo/edit#gid=0
"""

"""
Urteil besteht aus Hauptbegehren (normalerweise erste Ziffer), Nebenbegehren (in folgenden Ziffern) (Gerichtskosten, Rechtskosten, superprovisorische Mittel) und der formellen Mitteilung (letzte Ziffer) an die Vorinstanz und die Parteien (und evtl. andere).
Der Einfachheit halber wird nur das Hauptbegehren berücksichtigt.
Das Hauptbegehren ist aussagekräftiger

Zwei Strategien:
    1. Nur die erste Ziffer berücksichtigen (meistens Hauptbegehren):
        Vorteil: 
            - Mehrheitlich wird in der ersten Ziffer über das Hauptbegehren entschieden
            - Das Hauptbegehren ist meistens bei weitem wichtiger als die Nebenbegehren
        Problem:
            - Teilweise wird das Hauptbegehren erst in einer folgenden Ziffer entschieden (bspw. zwei Verfahren werden vereinigt)
    2. Alle Ziffern berücksichtigen:
        Vorteil:
            - Simplerer Code
            - Das Hauptbegehren wird sicher gefunden
        Problem:
            - Die Nebenbegehren verzerren das Resultat

Da das Problem der 1. Strategie wahrscheinlich seltener auftritt, wird Strategie 1 gewählt.

Hauptbegehren:
    - Gutheissung (approval) (materiell/inhaltlich): Das Begehren wird vollständig akzeptiert
    - Teilweise Gutheissung (partial_approval) (materiell/inhaltlich): Ein Teil des Begehrens wird akzeptiert
    - Abweisung (dismissal) (materiell/inhaltlich): Das Begehren wird vollständig abgewiesen
    - Teilweise Abweisung (partial_dismissal) (materiell/inhaltlich): Ein Teil des Begehrens wird abgewiesen
    - Nichteintreten (inadmissible) (formell): Das Gericht ist nicht zuständig für das Begehren, Formelle Mängel der Beschwerde 
    - Abschreibung (write_off) (formell): Gegenstandslosigkeit, Kein Grund für das Verfahren (der Entscheid wird nicht mehr benötigt, da bspw. die Parteien sich aussergerichtlich geeinigt haben oder da zwei Verfahren vereinigt werden)
    - Vereinigung (unification) (formell): When two cases are about the same thing, they will get merged into one.
    
Nebenbegehren:
    - Gerichtskosten
        - Nicht Erhoben 
        - Auferlegt
    - Kosten für Rechtspflege/Rechtsvertretung (Anwälte)
        - 
    - superprovisorische Mittel (aufschiebende Wirkung)
    - gibts noch anderes?

Formelle Mitteilung:
    - Vorinstanz
    - Parteien
    - Andere

"""

all_judgment_markers = {
    Language.DE: {
        Judgment.APPROVAL: ['aufgehoben', 'aufzuheben', 'gutgeheissen', 'gutzuheissen', 'In Gutheissung', 'schuldig erklärt', 'rechtmässig'],
        Judgment.PARTIAL_APPROVAL: ['teilweise gutgeheissen', 'teilweise gutzuheissen',
                                    'In teilweiser Gutheissung'],
        Judgment.DISMISSAL: ['abgewiesen', 'abzuweisen', 'erstinstanzliche Urteil wird bestätigt', 'freigesprochen'],
        Judgment.PARTIAL_DISMISSAL: ['abgewiesen, soweit darauf einzutreten ist',
                                     'abzuweisen, soweit darauf einzutreten ist',
                                     'abgewiesen, soweit auf sie einzutreten ist',
                                     'abzuweisen, soweit auf sie einzutreten ist'],
        Judgment.INADMISSIBLE: ['Nichteintreten', 'nicht eingetreten', 'nicht einzutreten',
                                'wird keine Folge geleistet', 'wird nicht eingegangen',
                                'soweit darauf einzutreten ist', 'soweit auf sie einzutreten ist'],
        Judgment.WRITE_OFF: ['abgeschrieben', 'abzuschreiben', 'erweist sich als gegenstandslos'],
        Judgment.UNIFICATION: [
            "werden vereinigt", "werden gemeinsam beurteilt", "werden nicht vereinigt"]
    },
    Language.FR: {
        Judgment.APPROVAL: ['admis', 'est annulé', 'Admet', 'admet'],
        Judgment.PARTIAL_APPROVAL: ['Admet partiellement',
                                    'partiellement admis',
                                    'admis dans la mesure où il est recevable',
                                    'admis dans la mesure où ils sont recevables'
                                    ],
        Judgment.DISMISSAL: ['rejeté', 'Rejette', 'écarté', 'rejette'],
        Judgment.PARTIAL_DISMISSAL: ['rejetés dans la mesure où ils sont recevables',
                                     'rejeté, dans la mesure où il est recevable',
                                     'rejeté dans la mesure où il est recevable',
                                     'rejeté dans la mesure de sa recevabilité'
                                     ],
        Judgment.INADMISSIBLE: ['N\'entre pas en matière', 'irrecevable', 'n\'est pas entré',
                                'pas pris en considération'],
        Judgment.WRITE_OFF: ['retrait', 'est radiée', 'sans objet', 'rayé', 'Raye'],
        Judgment.UNIFICATION: ['no entries yet for unification'],
    },
    Language.IT: {
        Judgment.APPROVAL: ['accolt',  # accolt o/i/a/e
                            'annullat'],  # annullat o/i/a/e
        Judgment.PARTIAL_APPROVAL: ['Nella misura in cui è ammissibile, il ricorso è parzialmente accolto',
                                    'In parziale accoglimento del ricorso'],
        Judgment.DISMISSAL: ['respint',  # respint o/i/a/e
                             ],
        Judgment.PARTIAL_DISMISSAL: ['Nella misura in cui è ammissibile, il ricorso è respinto',
                                     'Nella misura in cui è ammissibile, il ricorso di diritto pubblico è respinto',
                                     'Nella misura in cui è ammissibile, la domanda di revisione è respinta'],
        Judgment.INADMISSIBLE: ['inammissibil',  # inamissibil o/i/a/e
                                'irricevibil',  # irricevibil o/i/a/e
                                ],
        Judgment.WRITE_OFF: ['privo d\'oggetto', 'priva d\'oggetto', 'privo di oggetto', 'priva di oggetto',
                             'è stralciata dai ruoli a seguito del ritiro del ricorso',
                             'è stralciata dai ruoli in seguito al ritiro del ricorso',
                             'stralciata dai ruoli',  # maybe too many mistakes
                             'radiata dai ruoli',  # maybe too many mistakes
                             ],
        Judgment.UNIFICATION: ['sono congiunte'],
    }
}


def XX_SPIDER(rulings: str, namespace: dict) -> Optional[List[Judgment]]:
    """
    Extract judgment outcomes from the rulings
    :param rulings:     the string containing the rulings
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the list of judgments
    """
    # This is an example spider. Just copy this method and adjust the method name and the code to add your new spider.
    if namespace['language'] not in all_judgment_markers:
            message = f"This function is only implemented for the languages {list(all_judgment_markers.keys())} so far."
            raise ValueError(message)

    # make sure we don't have any nasty unicode problems
    rulings = clean_text(rulings)

    judgments = get_judgments(rulings, namespace)

    judgments = verify_judgments(judgments, rulings, namespace)
    
    return judgments

def ZH_Baurekurs(rulings: str, namespace: dict) -> Optional[List[Judgment]]:
    """
    Extract judgment outcomes from the rulings
    :param rulings:     the string containing the rulings
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the list of judgments
    """

    if namespace['language'] not in all_judgment_markers:
        message = f"This function is only implemented for the languages {list(all_judgment_markers.keys())} so far."
        raise ValueError(message)

    # make sure we don't have any nasty unicode problems
    rulings = clean_text(rulings)

    judgments = unnumbered_rulings(rulings, namespace)

    judgments = verify_judgments(judgments, rulings, namespace)
    
    return judgments

def BS_Omni(rulings: str, namespace: dict) -> Optional[List[Judgment]]:
    """
    Extract judgment outcomes from the rulings
    :param rulings:     the string containing the rulings
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the list of judgments
    """

    if namespace['language'] not in all_judgment_markers:
        message = f"This function is only implemented for the languages {list(all_judgment_markers.keys())} so far."
        raise ValueError(message)

    # make sure we don't have any nasty unicode problems
    rulings = clean_text(rulings)

    judgments = unnumbered_rulings(rulings, namespace)

    judgments = verify_judgments(judgments, rulings, namespace)
    
    return judgments
def LU_Gerichte(rulings: str, namespace: dict) -> Optional[List[Judgment]]:
    """
    Extract judgment outcomes from the rulings
    :param rulings:     the string containing the rulings
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the list of judgments
    """

    if namespace['language'] not in all_judgment_markers:
        message = f"This function is only implemented for the languages {list(all_judgment_markers.keys())} so far."
        raise ValueError(message)

    # make sure we don't have any nasty unicode problems
    rulings = clean_text(rulings)

    judgments = unnumbered_rulings(rulings, namespace)

    judgments = verify_judgments(judgments, rulings, namespace)
    
    return judgments

def JU_Gerichte(rulings: str, namespace: dict) -> Optional[List[Judgment]]:
    """
    Extract judgment outcomes from the rulings
    :param rulings:     the string containing the rulings
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the list of judgments
    """

    if namespace['language'] not in all_judgment_markers:
        message = f"This function is only implemented for the languages {list(all_judgment_markers.keys())} so far."
        raise ValueError(message)

    # make sure we don't have any nasty unicode problems
    rulings = clean_text(rulings)

    judgments = unnumbered_rulings(rulings, namespace)

    judgments = verify_judgments(judgments, rulings, namespace)
    
    return judgments

def GE_Gerichte(rulings: str, namespace: dict) -> Optional[List[Judgment]]:
    """
    Extract judgment outcomes from the rulings
    :param rulings:     the string containing the rulings
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the list of judgments
    """

    if namespace['language'] not in all_judgment_markers:
        message = f"This function is only implemented for the languages {list(all_judgment_markers.keys())} so far."
        raise ValueError(message)

    # make sure we don't have any nasty unicode problems
    rulings = clean_text(rulings)

    judgments = unnumbered_rulings(rulings, namespace)

    judgments = verify_judgments(judgments, rulings, namespace)
    
    return judgments

def CH_EDOEB(rulings: str, namespace: dict) -> Optional[List[Judgment]]:
    """
    Extract judgment outcomes from the rulings
    :param rulings:     the string containing the rulings
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the list of judgments
    """

    if namespace['language'] not in all_judgment_markers:
        message = f"This function is only implemented for the languages {list(all_judgment_markers.keys())} so far."
        raise ValueError(message)

    # make sure we don't have any nasty unicode problems
    rulings = clean_text(rulings)

    judgments = unnumbered_rulings(rulings, namespace)

    judgments = verify_judgments(judgments, rulings, namespace)

    return judgments


def UR_Gerichte(rulings: str, namespace: dict) -> Optional[List[Judgment]]:
    """
    Extract judgment outcomes from the rulings
    :param rulings:     the string containing the rulings
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the list of judgments
    """

    # Rather than extend and update the global markers, for this spider, specific markers are defined.
    all_judgment_markers = {
        Language.DE: {
            Judgment.APPROVAL: [r'Gutheissung der Beschwerde', r'Bejahung der Beschwerdelegimitation'],
            Judgment.PARTIAL_APPROVAL: [r'Teilweise Gutheissung der Beschwerde'],
            Judgment.DISMISSAL: [r'Abweisung der Beschwerde',
                                 r'Der Anzeige wird keine Folge gegeben',
                                 r'Verneinung der Beschwerdelegimitation',
                                 r'Abweisung der Verwaltungsgerichtsbeschwerde',
                                 r'Abweisung der [Vv]erwaltungsrechtlichen Klage',
                                 r'Abweisung des Gesuches um Wiederherstellung der Frist',
                                 r'In concreto Abweisung der Berufung der Dienstbarkeitsbelasteten'],
            Judgment.WRITE_OFF: [
                r'Abschreibung der Beschwerde vom Geschäftsprotokoll ']
        }
    }

    # In canton UR, de only
    if namespace['language'] != Language.DE:
        raise ValueError(
            f'This function is only implemented for {Language.DE} so far.')

    def getJudgments(text: str, judgment_markers, debug=False) -> Optional[List[Judgment]]:
        """
        Get the judgment outcomes based on a regex dictionary for a given section string.
        :param text:               the text string of the section, usually Section.RULINGS
        :param judgment_markers:   the regex dicttionary for the different judgment outcomes
        :param debug:              if set to True, prints debug output to console
        :return:                   the list of judgment outcomes
        """

        judgments = []
        for lang in judgment_markers:
            for judg in judgment_markers[lang]:
                for reg in (judgment_markers[lang])[judg]:
                    matches = re.finditer(reg, text, re.MULTILINE)
                    for num, match in enumerate(matches, start=1):
                        from_, to_, match_text = match.start(), match.end(), match.group()
                        judgments.append(judg)
                        if debug:
                            print(
                                f'{judg} ("{match.group()}") at {to_/len(text):.1%} of the section.')

        return judgments

    # find all judgments in the rulings
    judgments = getJudgments(rulings, all_judgment_markers)

    # validate
    if len(judgments) > 1:
        message = f"For the decision {namespace['html_url']} several rulings where found from the rulings: {rulings}"
        raise ValueError(message)
    if len(judgments) == 0:
        message = f"For the decision {namespace['html_url']} no main ruling was found from the rulings: {rulings}"
        raise ValueError(message)

    return judgments


def get_judgments(rulings: str, namespace: dict) -> set:
    """
    Get the judgment outcomes based on a rulings string and the given namespace context.
    :param rulings:     the rulings string
    :param namespace:   the context (metadata) from the court decision
    :return:            the set of judgment outcomes
    """
    judgments = set()

    judgment_markers = prepare_judgment_markers(
        all_judgment_markers, namespace)

    pattern = rf"{1}\.(.+?)(?:{2}\.|$)"
    romanPattern = rf"{int_to_roman(1)}\.(.+?)(?:{int_to_roman(2)}\.|$)"

    if (re.search(pattern, rulings) or re.search(romanPattern, rulings)):
        judgments = numbered_rulings(
            judgments, rulings, namespace, judgment_markers)
    return judgments

def verify_judgments(judgments, rulings, namespace):
    if not judgments:
        message = f"Found no judgment for the rulings \"{rulings}\" in the case {namespace['html_url']}. Please check!"
        raise ValueError(message)
    elif len(judgments) > 1:
        if Judgment.PARTIAL_APPROVAL in judgments:
            # if partial_approval is found, it will find approval as well
            judgments.discard(Judgment.APPROVAL)
        if Judgment.PARTIAL_DISMISSAL in judgments:
            # if partial_dismissal is found, it will find dismissal as well
            judgments.discard(Judgment.DISMISSAL)
    return judgments

def unnumbered_rulings(rulings: str, namespace: dict):
    judgments = set()
    judgment_markers = prepare_judgment_markers(
        all_judgment_markers, namespace)
    return iterate_Judgments(rulings, judgments, judgment_markers, False)


def numbered_rulings(judgments: set, rulings: str, namespace: dict, judgment_markers: dict):
    n = 1
    while len(judgments) == 0:
        try:
            ruling = get_nth_ruling(rulings, namespace, n)
            judgments = iterate_Judgments(
                ruling, judgments, judgment_markers, True)
            n += 1
        except ValueError:
            break
    return judgments


def iterate_Judgments(ruling: str, judgments: set, judgment_markers: dict, numberedRuling: bool) -> set:
    positions = []
    for judgment in Judgment:
        markers = judgment_markers[judgment]
        # if we don't do this, we get weird matching behaviour
        ruling = unicodedata.normalize('NFC', ruling)
        matching = re.search(markers, ruling)
        if matching:
            if numberedRuling:
                judgments.add(judgment)
            else:
                positions.append({"match": matching, "judgment": judgment})
    if not numberedRuling and positions:
        judgments = getFirstInstance(positions, judgments)
    return judgments

def getFirstInstance(positions: dict, judgments: set) -> set:
    firstInstance = positions[0]
    judgments = {firstInstance["judgment"]}
    for judgment in positions[1:]:
        position = firstInstance["match"].span()
        comparison = judgment["match"].span()
        if(comparison[0] < position[0]):
            firstInstance = judgment
            judgments = {firstInstance["judgment"]}
        elif(position[0] == comparison[0]):
            firstInstance = judgment
            judgments.add(firstInstance["judgment"])
    return judgments


def get_nth_ruling(rulings: str, namespace: dict, n: int) -> str:
    """
    Gets the nth ruling from the rulings string
    :param rulings:     the rulings string
    :param namespace:   the context (metadata) from the court decision
    :param n:           the nth ruling to be retrieved
    :return:            the string of the nth ruling
    """
    result = search_rulings(rulings, str(n), str(n + 1))
    if not result:
        # try with roman numerals
        result = search_rulings(rulings, int_to_roman(n), int_to_roman(n + 1))
        if not result:
            message = f"For the decision {namespace['html_url']} no main ruling was found from the rulings: {rulings}"
            raise ValueError(message)
    return result.group(1)


def search_rulings(rulings: str, start: str, end: str):
    """
    Search the rulings for start and end indicating the boundaries of the ruling to be found
    :param rulings: the rulings string
    :param start:   the string indicating the start of the ruling to be found
    :param end:     the string indicating the end of the ruling to be found
    :return:        the ruling between the start and end string
    """
    pattern = rf"{start}\.(.+?)(?:{end}\.|$)"
    return re.search(pattern, rulings)

def prepare_judgment_markers(all_judgment_markers: dict, namespace: dict) -> dict:
    judgment_markers = all_judgment_markers[namespace['language']]
    # combine multiple regex into one for each section due to performance reasons
    judgment_markers = dict(
        map(lambda kv: (kv[0], '|'.join(kv[1])), judgment_markers.items()))
    return judgment_markers
