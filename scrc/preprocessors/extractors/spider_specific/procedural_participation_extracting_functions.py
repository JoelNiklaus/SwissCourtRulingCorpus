from typing import Dict, List, Optional
import re
import json
from scrc.data_classes.legal_counsel import LegalCounsel
from scrc.data_classes.procedural_participation import ProceduralParticipation
from scrc.data_classes.proceedings_party import ProceedingsParty

from scrc.enums.gender import Gender
from scrc.enums.language import Language
from scrc.enums.legal_type import LegalType
from scrc.enums.section import Section

"""
This file is used to extract the parties from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
Overview of spiders still todo: https://docs.google.com/spreadsheets/d/1FZmeUEW8in4iDxiIgixY4g0_Bbg342w-twqtiIu8eZo/edit#gid=0
"""


def XX_SPIDER(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    # This is an example spider. Just copy this method and adjust the method name and the code to add your new spider.
    pass


def CH_BGer(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract lower courts from decisions of the Federal Supreme Court of Switzerland
    :param header:      the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict
    """
    header = sections[Section.HEADER]

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
        Gender.MALE: [r'Beschwerdeführer(?!in)', r'Beschwerdegegner(?!in)', r'recourant(?!e)', r'intimés?(?!e)',
                      r'ricorrente'],
        Gender.FEMALE: [r'Beschwerdeführerin', r'Beschwerdegegnerin', r'recourantes?', r'intimées?']
    }

    lawyer_representation = {
        Gender.MALE: [r'Rechtsanwalt', r'Fürsprecher(?!in)', r'Advokat(?!in)', r'avocats?(?!e)', r'dall\'avv\.',
                      r'l\'avv\.'],
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


def ZG_Verwaltungsgericht(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract the procedural participation from the Verwaltungsgericht of Zug.
    :param sections:    a dict containing the sections of a decision
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the procedural participation
    """

    header = sections[Section.HEADER]

    information_start_regex, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name = get_regex()

    header = get_participation_from_header(header, information_start_regex, namespace)
    party = get_procedural_participation(header, namespace, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name)

    return party.toJSON()


def ZH_Baurekurs(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract the procedural participation from the Baurekursgericht of Zurich.
    :param sections:    a dict containing the sections of a decision
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the procedural participation
    """

    header = sections[Section.HEADER]

    information_start_regex, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name = get_regex()

    header = get_participation_from_header(header, information_start_regex, namespace)
    party = get_procedural_participation(header, namespace, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name)

    return party.toJSON()


def ZH_Obergericht(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract the procedural participation from the Obergericht of Zurich.
    :param sections:    a dict containing the sections of a decision
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the procedural participation
    """

    header = sections[Section.HEADER]

    information_start_regex, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name = get_regex()

    header = get_participation_from_header(header, information_start_regex, namespace)
    party = get_procedural_participation(header, namespace, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name)

    return party.toJSON()


def ZH_Sozialversicherungsgericht(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract the procedural participation from the Sozialversicherungsgericht of Zurich.
    :param sections:    a dict containing the sections of a decision
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the procedural participation
    """

    header = sections[Section.HEADER]

    information_start_regex, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name = get_regex()

    header = get_participation_from_header(header, information_start_regex, namespace)
    party = get_procedural_participation(header, namespace, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name)

    return party.toJSON()


def ZH_Steuerrekurs(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract the procedural participation from the Steuerrekursgericht of Zurich.
    :param sections:    a dict containing the sections of a decision
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the procedural participation
    """

    header = sections[Section.HEADER]

    information_start_regex, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name = get_regex()

    header = get_participation_from_header(header, information_start_regex, namespace)
    party = get_procedural_participation(header, namespace, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name)

    return party.toJSON()


def ZH_Verwaltungsgericht(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract the procedural participation from the Verwaltungsgericht of Zurich.
    :param sections:    a dict containing the sections of a decision
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the procedural participation
    """

    header = sections[Section.HEADER]

    information_start_regex, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name = get_regex()

    header = get_participation_from_header(header, information_start_regex, namespace)
    party = get_procedural_participation(header, namespace, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name)

    return party.toJSON()


def get_regex():
    information_start_regex = r'Parteien|Verfahrensbeteiligte|[Ii]n Sachen'
    second_party_start_regex = [
        r'gegen',
    ]
    representation_start = {
        r'vertreten durch',
    }
    party_gender = {
        Gender.MALE: [r'Beschwerdeführer(?!in)', r'Beschwerdegegner(?!in)', r'Antragsteller(?!in)', r'Antragsgegner(?!in)', r'Rekurrent(?!in)', r'Rekursgegner(?!in)'],
        Gender.FEMALE: [r'Beschwerdeführerin', r'Beschwerdegegnerin', r'Antragstellerin', r'Antragsgegnerin', r'Rekurrentin', r'Rekursgegnerin']
    }
    lawyer_representation = {
        Gender.MALE: [r'Rechtsanwalt', r'Fürsprecher(?!in)', r'Advokat(?!in)'],
        Gender.FEMALE: [r'Rechtsanwältin', r'Fürsprecherin', r'Advokatin'],
        Gender.UNKNOWN: [r'RA']
    }
    lawyer_name = {
        Language.DE: r'((Dr\.\s)|(Prof\.\s))*[\w\séäöü\.]*?(?=(,)|($)| Gegen| und)'
    }

    representation_start = '|'.join(representation_start)
    for key in lawyer_representation:
        lawyer_representation[key] = '|'.join(lawyer_representation[key])
    for key in party_gender:
        party_gender[key] = '|'.join(party_gender[key])

    return information_start_regex, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name


    
def get_participation_from_header(header: str, information_start_regex: dict, namespace: dict) -> str:
    """
    Extract the portion of the header that contains the procedural participants.
    :param header:                      the header of a decision
    :param information_start_regex:     regex to find the start of the relevant section
    :param namespace:                   the namespace containing some metadata of the court decision
    :return:                            a string containing the relevant header part
    """

    start_pos = re.search(information_start_regex, header) or re.search(r'Gerichtsschreiber.*?\.', header)
    if start_pos:
        header = header[start_pos.span()[1]:]
    end_pos = {
        Language.DE: re.search(r'betreffend', header) or re.search(r'Sachverhalt', header) or re.search(r'Gegenstand', header) or re.search(r'gegen das Urteil', header)
    }
    end_pos = end_pos[namespace['language']]
    if end_pos:
        header = header[:end_pos.span()[0]]
    return header


def search_lawyers(text: str, lawyer_representation: dict, lawyer_name: dict, namespace: dict) -> List[LegalCounsel]:
    """
    Extract the legal counsel from a string.
    :param text:                    the part of the header containing the procedural participants
    :param lawyer_representation:   regex to find the lawyers
    :param lawyer_name:             regex to find the names of the lawyers
    :param namespace:               the namespace containing some metadata of the court decision
    :return:                        a list containing the legal counsel
    """

    lawyers: List[LegalCounsel] = []
    for (gender, current_regex) in lawyer_representation.items():
        pos = re.search(current_regex, text)
        if pos:
            lawyer = LegalCounsel('')
            lawyer.gender = Gender.UNKNOWN
            if not namespace['language'] == Language.IT:
                lawyer.gender = gender
            name_match = re.search(lawyer_name[namespace['language']], text[pos.span()[1]:])
            if name_match and not text[pos.span()[1]] == ',':
                lawyer.name = name_match.group()
            else:
                name_match = re.search(lawyer_name[namespace['language']], text[:pos.span()[0]])
                lawyer.name = name_match.group() if name_match else None
            lawyer.name = lawyer.name.strip()
            lawyer.legal_type = LegalType.NATURAL_PERSON
            lawyers.append(lawyer)

    return lawyers


def add_representation(text: str, representation_start: dict, lawyer_representation: dict, lawyer_name: dict, namespace: dict) -> List[LegalCounsel]:
    """
    Extract the legal representation from a string.
    :param text:                    the part of the header containing the legal representation
    :param representation_start:    regex to find the start of the legal representation
    :param lawyer_representation:   regex to find the lawyers
    :param lawyer_name:             regex to find the names of the lawyers
    :param namespace:               the namespace containing some metadata of the court decision
    :return:                        a list containing the legal counsel
    """
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
        lawyers = search_lawyers(current_text, lawyer_representation, lawyer_name, namespace)
        if lawyers:
            representations.extend(lawyers)
            continue

        name_match = re.search(r'[A-Z][\w\s\.\-\']*(?=\b)', current_text)
        if name_match:
            name = name_match.group()
            if name.startswith('Me'):
                lawyer = LegalCounsel(name[2:], legal_type=LegalType.NATURAL_PERSON)
                representations.append(lawyer)
                continue
            lawyer = LegalCounsel(name.strip(), legal_type=LegalType.LEGAL_ENTITY)
            if lawyer.gender == None:
                lawyer.gender = Gender.UNKNOWN
            representations.append(lawyer)
            continue
        name_match = re.search(r'[A-Z][\w\s\.\-\']*', current_text)
        if name_match:
            name = name_match.group()
            lawyer = LegalCounsel(name.strip(), legal_type=LegalType.LEGAL_ENTITY)
            if lawyer.gender == None:
                lawyer.gender = Gender.UNKNOWN
            representations.append(lawyer)
            continue
    representations = list(set(representations))  # remove duplicates
    return representations



def get_party(text: str, namespace: dict, party_gender: dict) -> List[ProceedingsParty]:
    """
    Extract the proceedings party from a string.
    :param text:            the part of the header containing the proceedings party
    :param namespace:       the namespace containing some metadata of the court decision
    :param party_gender:    regex to find the proceedings party
    :return:                a list containing the proceedings party
    """
    current_person = ProceedingsParty('')
    current_person.gender = Gender.UNKNOWN
    result: List[ProceedingsParty] = []
    try:
        current_person.name = re.search(r'[A-Z1-9].*?(?=(,)|(.$)| Beschwerde)', text).group().strip()
    except AttributeError:
        return result

    if re.match(r'[1-9IVX]+\.(?!_)', current_person.name):
        people_string = re.split(r'[1-9IVX]+\. ', text)
        for string in people_string[1:]:
            result.extend(get_party(string, namespace, party_gender))

        for idx in range(len(result)):
            result[idx].gender = Gender.UNKNOWN
        return result
    if re.match(r'([A-Z]\.)?[A-Z]\._$', current_person.name):
        for gender, current_regex in party_gender.items():
            if re.search(current_regex, text):
                current_person.gender = Gender.UNKNOWN
                if not namespace['language'] == Language.IT and not gender == None:
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



def get_procedural_participation(header: str, namespace: dict, second_party_start_regex: list, representation_start: dict, party_gender: dict, lawyer_representation: dict, lawyer_name: dict) -> ProceduralParticipation:
    """
    Extract the procedural participation from the header.
    :param header:                      the header of a decision
    :param namespace:                   the namespace containing some metadata of the court decision
    :param second_party_start_regex:    regex to find the start of the second party
    :param representation_start:        regex to find the the start of the legal representation
    :param party_gender:                regex to find the parties
    :param lawyer_representation:       regex to find the lawyers
    :param lawyer_name:                 regex to find the names of the lawyers
    :return:                            the procedural participation
    """
    header_parts = re.split('|'.join(second_party_start_regex), header)
    if len(header_parts) < 2:
        raise ValueError(f"({namespace['id']}): Header malformed for: {namespace['html_url']}")
    party = ProceduralParticipation()
    plaintiff_representation = add_representation(header_parts[0], representation_start, lawyer_representation, lawyer_name, namespace)
    defendant_representation = add_representation(header_parts[1], representation_start, lawyer_representation, lawyer_name, namespace)

    party.plaintiffs = get_party(header_parts[0], namespace, party_gender)
    party.defendants = get_party(header_parts[1], namespace, party_gender)

    for plaintiff in party.plaintiffs:
        plaintiff.legal_counsel = plaintiff_representation
    for defendant in party.defendants:
        defendant.legal_counsel = defendant_representation
    return party


