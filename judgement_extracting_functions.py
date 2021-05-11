from typing import Any, Optional


def CH_BGer(rulings: str, namespace: dict) -> Optional[str]:
    """
    IMPORTANT: So far, only German is supported!
    :param rulings:     the string containing the rulings
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict, None if not in German
    """
    if namespace['language'] != 'de':
        return None

    judgement_markers = {'approval': ['In Gutheissung', 'aufgehoben', 'gutgeheissen', 'gutzuheissen'],
                         'partial approval': ['teilweise gutgeheissen'],
                         'dismissal': ['abgewiesen'],
                         'partial dismissal': ['abgewiesen, soweit darauf einzutreten ist'],
                         'formal problems': ['nicht eingetreten', 'als gegenstandslos abgeschrieben', 'Nichteintreten']}

    judgements = set()
    for judgement, markers in judgement_markers.items():
        for marker in markers:
            if marker in rulings:
                judgements.add(judgement)

    if not judgements:
        return None  # no judgements found
    elif len(judgements) > 1:
        message = f"Found more than one judgement ({judgements}) for the case {namespace['html_url']}. Please check!"
        raise ValueError(message)

    return judgements.pop()


def CH_BGE(rulings: str, namespace: dict) -> Optional[dict]:
    return CH_BGer(rulings, namespace)
