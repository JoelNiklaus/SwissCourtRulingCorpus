from typing import Optional, List

from scrc.utils.main_utils import string_contains_one_of_list


def CH_BGer(rulings: str, namespace: dict) -> Optional[List[str]]:
    """
    IMPORTANT: So far, only German is supported!
    :param rulings:     the string containing the rulings
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict, None if not in German
    """
    if namespace['language'] != 'de':
        return None

    judgement_markers = {'approval': ['In Gutheissung', 'aufgehoben', 'gutgeheissen', 'gutzuheissen'],
                         'partial_approval': ['teilweise gutgeheissen', 'In teilweiser Gutheissung'],
                         'dismissal': ['abgewiesen'],
                         'partial_dismissal': ['abgewiesen, soweit darauf einzutreten ist',
                                               'abzuweisen, soweit darauf einzutreten ist'],
                         'not_admitted': ['nicht eingetreten', 'als gegenstandslos abgeschrieben',
                                          # TODO sollte dies nicht abschreibung sein?
                                          'Nichteintreten', 'wird keine Folge geleistet'],
                         # 'soweit darauf einzutreten ist'],
                         'write_off': ['abgeschrieben']}

    judgements = set()
    for judgement, markers in judgement_markers.items():
        if string_contains_one_of_list(rulings, markers):
            judgements.add(judgement)

    if not judgements:
        message = f"Found no judgement for the case {namespace['html_url']}. Please check!"
        raise ValueError(message)
    elif len(judgements) > 1:
        if "partial_approval" in judgements:
            judgements.discard("approval")  # if partial_approval is found, it will find approval as well
        if "partial_dismissal" in judgements:
            judgements.discard("dismissal")  # if partial_dismissal is found, it will find dismissal as well

    if len(judgements) > 1:
        message = f"Found multiple judgements {judgements} for the case {namespace['html_url']}. Please check!"
        raise ValueError(message)

    return list(judgements)


def CH_BGE(rulings: str, namespace: dict) -> Optional[List[str]]:
    return CH_BGer(rulings, namespace)
