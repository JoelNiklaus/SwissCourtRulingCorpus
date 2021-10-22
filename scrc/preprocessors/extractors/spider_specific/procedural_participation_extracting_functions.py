from typing import Dict, List, Optional
import re
import json
from scrc.data_classes.legal_counsel import LegalCounsel
from scrc.data_classes.procedural_participation import ProceduralParticipation
from scrc.data_classes.proceedings_party import ProceedingsParty

from scrc.enums.gender import Gender
from scrc.enums.language import Language
from scrc.enums.legal_type import LegalType

"""
This file is used to extract the parties from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
Overview of spiders still todo: https://docs.google.com/spreadsheets/d/1FZmeUEW8in4iDxiIgixY4g0_Bbg342w-twqtiIu8eZo/edit#gid=0
"""

def XX_SPIDER(header: str, namespace: dict) -> Optional[str]:
    # This is an example spider. Just copy this method and adjust the method name and the code to add your new spider.
    pass

def CH_BGer(header: str, namespace: dict) -> Optional[str]:
    """
    Extract lower courts from decisions of the Federal Supreme Court of Switzerland
    :param header:      the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict
    """

    information_start_regex = r'Parteien|Verfahrensbeteiligte|[Ii]n Sachen|Parties|Participants à la procédure|formée? par|[Dd]ans la cause|Parti|Partecipanti al procedimento|Visto il ricorso.*?da'
    second_party_start_regex = [
        r'gegen',
        r'contre',
        r'(?<=,) et',
        r'contro(?! l[ao] (?:decisione|sentenza|risoluzione|scritto))',
        r'contro l.*?che (?:l[oai] )?oppone (?:(?:il|l[oai]) ricorrente)?'
    ]
    representation_start = {
        r'vertreten durch',
        r'représentée? par',
        r'p\.a\.',
        r'patrocinat[oia]',
        r'rappresentat[oia]',
        r'presso'
    }

    party_gender = {
        Gender.MALE: [r'Beschwerdeführer(?!in)', r'Beschwerdegegner(?!in)', r'recourant(?!e)', r'intimés?(?!e)', r'ricorrente'],
        Gender.FEMALE: [r'Beschwerdeführerin', r'Beschwerdegegnerin', r'recourantes?', r'intimées?']
    }

    lawyer_representation = {
        Gender.MALE: [r'Rechtsanwalt', r'Fürsprecher(?!in)', r'Advokat(?!in)', r'avocats?(?!e)', r'dall\'avv\.', r'l\'avv\.'],
        Gender.FEMALE: [r'Rechtsanwältin', r'Fürsprecherin', r'Advokatin', r'avocates?']
    }

    lawyer_name = {
        Language.DE: r'((Dr\.\s)|(Prof\.\s))*[\w\séäöü\.]*?(?=(,)|(.$)|. Gegen| und)',
        Language.FR: r'(?<=Me\s)[\w\séèäöü\.\-]*?(?=,| et)|(?<=Mes\s)[\w\séèäöü\.\-]*?(?=,| et)|(?<=Maître\s)[\w\séèäöü\.\-]*?(?=,| et)',
        Language.IT: r'(lic\.?\s?|iur\.?\s?|dott\.\s?)*[A-Z].*?(?=,)'
    }

    start_pos = re.search(information_start_regex, header) or re.search(r'Gerichtsschreiber.*?\.', header) or re.search(
        r'[Gg]reffi[eè]re?.*?\S{2,}?\.', header)
    if start_pos:
        header = header[start_pos.span()[1]:]
    end_pos = {
        Language.DE: re.search(r'(?<=Beschwerdegegnerin).+?', header) or re.search(r'(?<=Beschwerdegegner).+?',
                                                                            header) or re.search(r'Gegenstand',
                                                                                                 header) or re.search(
            r'A\.\- ', header) or re.search(r'gegen das Urteil', header),
        Language.FR: re.search(r'Objet', header) or re.search(r'Vu', header),
        Language.IT: re.search(r'Oggetto', header)
    }
    end_pos = end_pos[namespace['language']]
    if end_pos:
        header = header[:end_pos.span()[0]]

    representation_start = '|'.join(representation_start)
    for key in lawyer_representation:
        lawyer_representation[key] = '|'.join(lawyer_representation[key])
    for key in party_gender:
        party_gender[key] = '|'.join(party_gender[key])

    def search_lawyers(text: str) -> List[LegalCounsel]:
        lawyers: List[LegalCounsel] = []
        for (gender, current_regex) in lawyer_representation.items():
            pos = re.search(current_regex, text)
            if pos:
                lawyer = LegalCounsel()
                if not namespace['language'] == Language.IT:
                    lawyer.gender = gender
                name_match = re.search(lawyer_name[namespace['language']], text[pos.span()[1]:])
                if name_match and not text[pos.span()[1]] == ',':
                    lawyer.name = name_match.group()
                else:
                    name_match = re.search(lawyer_name[namespace['language']], text[:pos.span()[0]])
                    lawyer.name = name_match.group() if name_match else None
                lawyer.legal_type = LegalType.NATURAL_PERSON
                lawyers.append(lawyer)

        return lawyers

    def add_representation(text: str) -> List[LegalCounsel]:
        representations = []
        start_positions = tuple(re.finditer(representation_start, text))
        if not start_positions:
            return []

        for match_index in range(len(start_positions)):
            start_pos = start_positions[match_index].span()[1]
            if match_index + 1 < len(start_positions):
                end_pos = start_positions[match_index + 1].span()[0]
            else:
                end_pos = len(text)
            current_text = text[start_pos:end_pos]
            lawyers = search_lawyers(current_text)
            if lawyers:
                representations.extend(lawyers)
                continue

            name_match = re.search(r'[A-Z][\w\s\.\-\']*(?=,)', current_text)
            if name_match:
                name = name_match.group()
                if name.startswith('Me'):
                    lawyer = LegalCounsel(name[2:], legal_type=LegalType.NATURAL_PERSON)
                    representations.append(lawyer)
                    continue
                lawyer = LegalCounsel(name, legal_type=LegalType.LEGAL_ENTITY)
                representations.append(lawyer)
                continue
            name_match = re.search(r'[A-Z][\w\s\.\-\']*', current_text)
            if name_match:
                name = name_match.group()
                lawyer = LegalCounsel(name, legal_type=LegalType.LEGAL_ENTITY)
                representations.append(lawyer)
                continue
        representations = list(set(representations))  # remove duplicates
        return representations

    def get_party(text: str) -> List[ProceedingsParty]:
        current_person = ProceedingsParty()
        result: List[ProceedingsParty] = []
        try:
            current_person.name = re.search(r'[A-Z1-9].*?(?=(,)|(.$)| Beschwerde)', text).group().strip()
        except AttributeError:
            return result

        if re.match(r'[1-9IVX]+\.(?!_)', current_person.name):
            people_string = re.split(r'[1-9IVX]+\. ', text)
            for string in people_string[1:]:
                result.extend(get_party(string))

            for idx in range(len(result)):
                result[idx].gender = None
            return result
        if re.match(r'([A-Z]\.)?[A-Z]\._$', current_person.name):
            for gender, current_regex in party_gender.items():
                if re.search(current_regex, text):
                    if not namespace['language'] == Language.IT:
                        current_person.gender = gender
                    current_person.legal_type = LegalType.NATURAL_PERSON
                    result.append(current_person)
                    return result
            current_person.legal_type = LegalType.NATURAL_PERSON
            result.append(current_person)
            return result
        current_person.legal_type = LegalType.LEGAL_ENTITY
        result.append(current_person)
        return result

    header_parts = re.split('|'.join(second_party_start_regex), header)
    if len(header_parts) < 2:
        raise ValueError(f"({namespace['id']}): Header malformed for: {namespace['html_url']}")
    party = ProceduralParticipation()
    plaintiff_representation = add_representation(header_parts[0])
    defendant_representation = add_representation(header_parts[1])

    party.plaintiffs = get_party(header_parts[0])
    party.defendants = get_party(header_parts[1])

    for plaintiff in party.plaintiffs:
        plaintiff.legal_counsel = plaintiff_representation
    for defendant in party.defendants:
        defendant.legal_counsel = defendant_representation
    return party.toJSON()

# This needs special care
# def CH_BGE(rulings: str, namespace: dict) -> Optional[List[str]]:
#    return CH_BGer(rulings, namespace)
