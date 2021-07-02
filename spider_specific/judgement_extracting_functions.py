from typing import Optional, List

import re
from scrc.utils.main_utils import string_contains_one_of_list, clean_text, int_to_roman

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

judgement_markers = {
    'de': {
        'approval': ['aufgehoben', 'aufzuheben', 'gutgeheissen', 'gutzuheissen', 'In Gutheissung'],
        'partial_approval': ['teilweise gutgeheissen', 'teilweise gutzuheissen',
                             'In teilweiser Gutheissung'],
        'dismissal': ['abgewiesen', 'abzuweisen'],
        'partial_dismissal': ['abgewiesen, soweit darauf einzutreten ist',
                              'abzuweisen, soweit darauf einzutreten ist',
                              'abgewiesen, soweit auf sie einzutreten ist',
                              'abzuweisen, soweit auf sie einzutreten ist'],
        'inadmissible': ['Nichteintreten', 'nicht eingetreten', 'nicht einzutreten',
                         'wird keine Folge geleistet', 'wird nicht eingegangen',
                         'soweit darauf einzutreten ist', 'soweit auf sie einzutreten ist'],
        'write_off': ['abgeschrieben', 'abzuschreiben', 'erweist sich als gegenstandslos'],
        'unification': ["werden vereinigt", "werden gemeinsam beurteilt", "werden nicht vereinigt"]
    },
    'fr': {
        'approval': ['admis', 'est annulé', 'Admet'],
        'partial_approval': ['Admet partiellement',
                             'partiellement admis',
                             'admis dans la mesure où il est recevable',
                             'admis dans la mesure où ils sont recevables'
                             ],
        'dismissal': ['rejeté', 'Rejette', 'écarté'],
        'partial_dismissal': ['rejetés dans la mesure où ils sont recevables',
                              'rejeté, dans la mesure où il est recevable',
                              'rejeté dans la mesure où il est recevable',
                              'rejeté dans la mesure de sa recevabilité'
                              ],
        'inadmissible': ['N\'entre pas en matière', 'irrecevable', 'n\'est pas entré', 'pas pris en considération'],
        'write_off': ['retrait', 'est radiée', 'sans objet', 'rayé', 'Raye'],
        'unification': [],
    },
    'it': {
        'approval': ['accolt',  # accolt o/i/a/e
                     'annullat'],  # annullat o/i/a/e
        'partial_approval': ['Nella misura in cui è ammissibile, il ricorso è parzialmente accolto',
                             'In parziale accoglimento del ricorso'],
        'dismissal': ['respint',  # respint o/i/a/e
                      'irricevibil',  # irricevibil o/i/a/e
                      ],
        'partial_dismissal': ['Nella misura in cui è ammissibile, il ricorso è respinto',
                              'Nella misura in cui è ammissibile, il ricorso di diritto pubblico è respinto',
                              'Nella misura in cui è ammissibile, la domanda di revisione è respinta'],
        'inadmissible': ['inammissibil'],  # inamissibil o/i/a/e
        'write_off': ['privo d\'oggetto', 'priva d\'oggetto', 'privo di oggetto', 'priva di oggetto',
                      'è stralciata dai ruoli a seguito del ritiro del ricorso',
                      'è stralciata dai ruoli in seguito al ritiro del ricorso',
                      'stralciata dai ruoli',  # maybe too many mistakes
                      'radiata dai ruoli',  # maybe too many mistakes
                      ],
        'unification': ['sono congiunte'],
    }
}

"""
This file is used to extract the judgement outcomes from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
"""


def get_judgements(rulings: str, namespace: dict) -> set:
    """
    Get the judgment outcomes based on a rulings string and the given namespace context.
    :param rulings:     the rulings string
    :param namespace:   the context (metadata) from the court decision
    :return:            the set of judgment outcomes
    """
    judgements = set()

    n = 1
    while len(judgements) == 0:
        try:
            # Only look at main ruling (the first one) because it is by far the most important one for the case
            main_ruling = get_nth_ruling(rulings, namespace, n)

            for judgement, markers in judgement_markers[namespace['language']].items():
                # TODO maybe we should change to regex search here to make it easier with declinations in fr and it
                if string_contains_one_of_list(main_ruling, markers):
                    judgements.add(judgement)
            n = n + 1
        except ValueError:
            break
    return judgements


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


def CH_BGer(rulings: str, namespace: dict) -> Optional[List[str]]:
    """
    Extract judgement outcomes from the Federal Supreme Court of Switzerland
    :param rulings:     the string containing the rulings
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict, None if not in German
    """

    if namespace['language'] not in judgement_markers:
        message = f"This function is only implemented for the languages {list(judgement_markers.keys())} so far."
        raise ValueError(message)

    # make sure we don't have any nasty unicode problems
    rulings = clean_text(rulings)

    judgements = get_judgements(rulings, namespace)

    if not judgements:
        message = f"Found no judgement for the rulings \"{rulings}\" in the case {namespace['html_url']}. Please check!"
        raise ValueError(message)
    elif len(judgements) > 1:
        if "partial_approval" in judgements:
            # if partial_approval is found, it will find approval as well
            judgements.discard("approval")
        if "partial_dismissal" in judgements:
            # if partial_dismissal is found, it will find dismissal as well
            judgements.discard("dismissal")

    return list(judgements)

# This needs special care
# def CH_BGE(rulings: str, namespace: dict) -> Optional[List[str]]:
#    return CH_BGer(rulings, namespace)
