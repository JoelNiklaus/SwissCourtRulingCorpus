from collections import OrderedDict
from typing import Optional, List

import re
from scrc.utils.main_utils import string_contains_one_of_list, clean_text

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
    - Nichteintreten (not_admitted) (formell): Das Gericht ist nicht zuständig für das Begehren, Formelle Mängel der Beschwerde 
    - Abschreibung (write_off) (formell): Gegenstandslosigkeit, Kein Grund für das Verfahren (der Entscheid wird nicht mehr benötigt, da bspw. die Parteien sich aussergerichtlich geeinigt haben oder da zwei Verfahren vereinigt werden)

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

judgement_markers = {'de': {}, 'fr': {}, 'it': {}}
judgement_markers['de'] = {'approval': ['aufgehoben', 'aufzuheben', 'gutgeheissen', 'gutzuheissen', 'In Gutheissung'],
                           'partial_approval': ['teilweise gutgeheissen', 'teilweise gutzuheissen',
                                                'In teilweiser Gutheissung'],
                           'dismissal': ['abgewiesen', 'abzuweisen'],
                           'partial_dismissal': ['abgewiesen, soweit darauf einzutreten ist',
                                                 'abzuweisen, soweit darauf einzutreten ist',
                                                 'abgewiesen, soweit auf sie einzutreten ist',
                                                 'abzuweisen, soweit auf sie einzutreten ist'],
                           'not_admitted': ['Nichteintreten', 'nicht eingetreten', 'wird keine Folge geleistet',
                                            'wird nicht eingegangen',
                                            'soweit darauf einzutreten ist', 'soweit auf sie einzutreten ist'],
                           'write_off': ['abgeschrieben', 'abzuschreiben', 'erweist sich als gegenstandslos']}


def CH_BGer(rulings: str, namespace: dict) -> Optional[List[str]]:
    """
    IMPORTANT: So far, only German is supported!
    :param rulings:     the string containing the rulings
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict, None if not in German
    """
    # TODO extend to fr and it
    if namespace['language'] != 'de':
        return None

    rulings = clean_text(rulings)  # make sure we don't have any nasty unicode problems

    # Only look at main ruling (the first one) because it is by far the most important one for the case
    main_ruling = get_nth_ruling(rulings, namespace, 1)

    # When two cases are about the same thing, they will get merged into one. The main ruling is the second one in that case
    combination = ["werden vereinigt", "werden gemeinsam beurteilt", "werden nicht vereinigt"]
    if string_contains_one_of_list(main_ruling, combination):
        main_ruling = get_nth_ruling(rulings, namespace, 2)

    judgements = set()
    for judgement, markers in judgement_markers[namespace['language']].items():
        if string_contains_one_of_list(main_ruling, markers):
            judgements.add(judgement)

    if not judgements:
        message = f"Found no judgement for the rulings {rulings} in the case {namespace['html_url']}. Please check!"
        raise ValueError(message)
    elif len(judgements) > 1:
        if "partial_approval" in judgements:
            judgements.discard("approval")  # if partial_approval is found, it will find approval as well
        if "partial_dismissal" in judgements:
            judgements.discard("dismissal")  # if partial_dismissal is found, it will find dismissal as well

    # We allow multiple judgements in all combinations
    # if len(judgements) > 1:
    #    message = f"Found multiple judgements {judgements} for the case {namespace['html_url']}. Please check!"
    #    raise ValueError(message)

    return list(judgements)


def get_nth_ruling(rulings, namespace, n):
    result = search_rulings(rulings, str(n), str(n + 1))
    if not result:
        result = search_rulings(rulings, int_to_roman(n), int_to_roman(n + 1))  # try with roman numerals
        if not result:
            raise ValueError(
                f"For the decision {namespace['html_url']} no main ruling was found from the rulings: {rulings}")
    return result.group(1)


def search_rulings(rulings, start_str, end_str):
    pattern = rf"{start_str}\.(.+?)(?:{end_str}\.|$)"
    return re.search(pattern, rulings)


def int_to_roman(num):
    roman = OrderedDict()
    roman[1000] = "M"
    roman[900] = "CM"
    roman[500] = "D"
    roman[400] = "CD"
    roman[100] = "C"
    roman[90] = "XC"
    roman[50] = "L"
    roman[40] = "XL"
    roman[10] = "X"
    roman[9] = "IX"
    roman[5] = "V"
    roman[4] = "IV"
    roman[1] = "I"

    def roman_num(num):
        for r in roman.keys():
            x, y = divmod(num, r)
            yield roman[r] * x
            num -= (r * x)
            if num <= 0:
                break

    return "".join([a for a in roman_num(num)])


def CH_BGE(rulings: str, namespace: dict) -> Optional[List[str]]:
    return CH_BGer(rulings, namespace)
