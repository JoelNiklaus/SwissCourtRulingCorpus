from typing import Dict, List, Optional
import re
import json
from scrc.data_classes.person import Person
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






def ZG_Verwaltungsgericht(header: str, namespace: dict) -> Optional[str]:
    """
    Extract procedural participation from the Verwaltungsgericht of Zug
    :param header:      the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the procedural participation
    """

    information_start_regex, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name = get_regex()

    header = get_participation_from_header(header, information_start_regex, namespace)
    party = get_procedural_participation(header, namespace, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name)

    return party.toJSON()

def ZH_Baurekurs(header: str, namespace: dict) -> Optional[str]:
    """
    Extract procedural participation from the Baurekursgericht of Zurich
    :param header:      the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the procedural participation
    """

    information_start_regex, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name = get_regex()

    header = get_participation_from_header(header, information_start_regex, namespace)
    party = get_procedural_participation(header, namespace, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name)

    return party.toJSON()

def ZH_Obergericht(header: str, namespace: dict) -> Optional[str]:
    """
    Extract procedural participation from the Obergericht of Zurich
    :param header:      the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the procedural participation
    """

    information_start_regex, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name = get_regex()

    header = get_participation_from_header(header, information_start_regex, namespace)
    party = get_procedural_participation(header, namespace, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name)

    return party.toJSON()

def ZH_Sozialversicherungsgericht(header: str, namespace: dict) -> Optional[str]:
    """
    Extract procedural participation from the Baurekursgericht of Zurich
    :param header:      the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the procedural participation
    """

    information_start_regex, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name = get_regex()

    header = get_participation_from_header(header, information_start_regex, namespace)
    party = get_procedural_participation(header, namespace, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name)

    return party.toJSON()

def ZH_Steuerrekurs(header: str, namespace: dict) -> Optional[str]:
    """
    Extract procedural participation from the Baurekursgericht of Zurich
    :param header:      the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the procedural participation
    """

    information_start_regex, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name = get_regex()

    header = get_participation_from_header(header, information_start_regex, namespace)
    party = get_procedural_participation(header, namespace, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name)

    return party.toJSON()

def ZH_Verwaltungsgericht(header: str, namespace: dict) -> Optional[str]:
    """
    Extract procedural participation from the Baurekursgericht of Zurich
    :param header:      the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the procedural participation
    """

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
        Language.DE: r'((Dr\.\s)|(Prof\.\s))*[\w\séäöü\.]*?(?=(,)|(.$)|. Gegen| und)'
    }

    representation_start = '|'.join(representation_start)
    for key in lawyer_representation:
        lawyer_representation[key] = '|'.join(lawyer_representation[key])
    for key in party_gender:
        party_gender[key] = '|'.join(party_gender[key])

    return information_start_regex, second_party_start_regex, representation_start, party_gender, lawyer_representation, lawyer_name

    
def get_participation_from_header(header: str, information_start_regex: dict, namespace: dict) -> str:
    start_pos = re.search(information_start_regex, header) or re.search(r'Gerichtsschreiber.*?\.', header)
    if start_pos:
        header = header[start_pos.span()[1]:]
    end_pos = {
        Language.DE: re.search(r'betreffend', header) or re.search(r'Sachverhalt', header) or re.search(r'(?<=Beschwerdegegnerin).+?', header) or re.search(r'(?<=Beschwerdegegner).+?', header) or re.search(r'Gegenstand', header) or re.search(r'A\.\- ', header) or re.search(r'gegen das Urteil', header)
    }
    end_pos = end_pos[namespace['language']]
    if end_pos:
        header = header[:end_pos.span()[0]]
    return header


def search_lawyers(text: str, lawyer_representation: dict, lawyer_name: dict, namespace: dict) -> List[LegalCounsel]:
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
    # uncomment to debug
    # print(party)
    return party





def testing():
    """
    This function tests the extracting functions checking if their output is correct given a test header (that was copied from the section splitting output)
    """
    ZG_Verwaltungsgericht_test_header = ['Normal.dot', 'VERWALTUNGSGERICHT DES KANTONS ZUG', 'SOZIALVERSICHERUNGSRECHTLICHE KAMMER', 'Mitwirkende Richter: lic. iur. Adrian Willimann, Vorsitz lic. iur. Jacqueline Iten-Staub und Dr. iur. Matthias Suter Gerichtsschreiber: MLaw Patrick Trütsch', 'U R T E I L vom 18. Juni 2020 [rechtskräftig] gemäss § 29 der Geschäftsordnung', 'in Sachen', 'A._ Beschwerdeführer vertreten durch B._ AG', 'gegen', 'Ausgleichskasse Zug, Baarerstrasse 11, Postfach, 6302 Zug Beschwerdegegnerin', 'betreffend', 'Ergänzungsleistungen (hypothetisches Erwerbseinkommen)', 'S 2019 121', '2', 'Urteil S 2019 121']

    ZH_Steuerrekurs_test_header = ['Endentscheid Kammer', 'Steuerrekursgericht des Kantons Zürich', '2. Abteilung', '2 DB.2017.240 2 ST.2017.296', 'Entscheid', '5. Februar 2019', 'Mitwirkend:', 'Abteilungspräsident Christian Mäder, Steuerrichterin Micheline Roth, Steuerrichterin Barbara Collet und Gerichtsschreiber Hans Heinrich Knüsli', 'In Sachen', '1. A, 2. B,', 'Beschwerdeführer/ Rekurrenten, vertreten durch C AG,', 'gegen', '1. Schw eizer ische E idgenossenschaf t , Beschwerdegegnerin, 2. Staat Zür ich , Rekursgegner, vertreten durch das kant. Steueramt, Division Konsum, Bändliweg 21, Postfach, 8090 Zürich,', 'betreffend', 'Direkte Bundessteuer 2012 sowie Staats- und Gemeindesteuern 2012', '- 2 -', '2 DB.2017.240 2 ST.2017.296']

    ZH_Baurekurs_test_header = ['BRGE Nr. 0/; GUTH vom', 'Baurekursgericht des Kantons Zürich', '2. Abteilung', 'G.-Nr. R2.2011.00160 BRGE II Nr. 0049/2012', 'Entscheid vom 20. März 2012', 'Mitwirkende Abteilungsvizepräsident Emil Seliner, Baurichter Peter Rütimann,  Adrian Bergmann, Gerichtsschreiber Robert Durisch', 'in Sachen Rekurrentin', 'Hotel Uto Kulm AG, Gratstrasse, 8143 Stallikon', 'vertreten durch Rechtsanwalt Dr. iur. Christof Truniger, Metzgerrainle 9, Postfach 5024, 6000 Luzern 5', 'gegen Rekursgegnerinnen', '1. Bau- und Planungskommission Stallikon, 8143 Stallikon 2. Baudirektion Kanton Zürich, Walchetor, Walcheplatz 2, Postfach,', '8090 Zürich', 'betreffend Bau- und Planungskommissionsbeschluss vom 24. August 2011 und Ver-', 'fügung der Baudirektion Kanton Zürich Nr. BVV 06.0429_1 vom 8. Juli 2011; Verweigerung der nachträglichen Baubewilligung für Aussen- und Turmbeleuchtung Uto Kulm (Neubeurteilung), Kat.-Nr. 1032, Gratstrasse, Hotel-Restaurant Uto Kulm, Üetliberg / Stallikon _', 'R2.2011.00160 Seite 2']

    ZH_Obergericht_test_header = ['Urteil', 'Handelsgericht des Kantons Zürich', 'Geschäfts-Nr.: HG150139-O U/jc', 'Mitwirkend: Oberrichter Dr. George Daetwyler, Präsident, und Oberrichterin Dr.', 'Claudia Bühler, die Handelsrichter Prof. Dr. Othmar Strasser, Peter', 'Leutenegger und Ursula Mengelt sowie die Gerichtsschreiberin', 'Kerstin Habegger', 'Urteil vom 7. Dezember 2015', 'in Sachen', 'A._ Genossenschaft,', 'Klägerin', 'vertreten durch Rechtsanwältin Dr. iur. X1._', 'vertreten durch Rechtsanwalt Dr. iur. X2._', 'gegen', 'B._,', 'Beklagter']

    ZH_Verwaltungsgericht_test_header = ['Verwaltungsgericht des Kantons Zürich 4. Abteilung', 'VB.2020.00452', 'Urteil', 'der 4. Kammer', 'vom 24. September 2020', 'Mitwirkend: Abteilungspräsidentin Tamara Nüssle (Vorsitz), Verwaltungsrichter Reto Häggi Furrer, Verwaltungsrichter Martin Bertschi, Gerichtsschreiber David Henseler.', 'In Sachen', 'A, vertreten durch RA B,', 'Beschwerdeführerin,', 'gegen', 'Migrationsamt des Kantons Zürich,', 'Beschwerdegegner,', 'betreffend vorzeitige Erteilung der Niederlassungsbewilligung,']

    ZH_Sozialversicherungsgericht_test_header = ['Sozialversicherungsgerichtdes Kantons ZürichIV.2014.00602', 'II. Kammer', 'Sozialversicherungsrichter Mosimann, Vorsitzender', 'Sozialversicherungsrichterin Käch', 'Sozialversicherungsrichterin Sager', 'Gerichtsschreiberin Kudelski', 'Urteil vom 11. August 2015', 'in Sachen', 'X._', 'Beschwerdeführerin', 'vertreten durch Rechtsanwalt Dr. Kreso Glavas', 'Advokatur Glavas AG', 'Markusstrasse 10, 8006 Zürich', 'gegen', 'Sozialversicherungsanstalt des Kantons Zürich, IV-Stelle', 'Röntgenstrasse 17, Postfach, 8087 Zürich', 'Beschwerdegegnerin', 'weitere Verfahrensbeteiligte:', 'Personalvorsorgestiftung der Y._', 'Beigeladene']


    ZG_Verwaltungsgericht_test_string = ', '.join(map(str, ZG_Verwaltungsgericht_test_header))
    ZH_Steuerrekurs_test_string = ', '.join(map(str, ZH_Steuerrekurs_test_header))
    ZH_Baurekurs_test_string = ', '.join(map(str, ZH_Baurekurs_test_header))
    ZH_Obergericht_test_string = ', '.join(map(str, ZH_Obergericht_test_header))
    ZH_Verwaltungsgericht_test_string = ', '.join(map(str, ZH_Verwaltungsgericht_test_header))
    ZH_Sozialversicherungsgericht_test_string = ', '.join(map(str, ZH_Sozialversicherungsgericht_test_header))

    namespace = {'language' : Language.DE}

    zg_vg_json = ZG_Verwaltungsgericht(ZG_Verwaltungsgericht_test_string, namespace)
    zg_vg = json.loads(zg_vg_json)
    assert zg_vg['plaintiffs'][0]['legal_counsel'][0]['name'] == 'B._ AG'
    assert zg_vg['plaintiffs'][0]['legal_counsel'][0]['legal_type'] == 'legal entity'

    zh_sr_json = ZH_Steuerrekurs(ZH_Steuerrekurs_test_string, namespace)
    zh_sr = json.loads(zh_sr_json)
    assert zh_sr['defendants'][0]['legal_counsel'][0]['name'] == 'Steueramt'
    assert zh_sr['defendants'][0]['legal_counsel'][0]['legal_type'] == 'legal entity'
    assert zh_sr['plaintiffs'][0]['legal_counsel'][0]['name'] == 'C AG'
    assert zh_sr['plaintiffs'][0]['legal_counsel'][0]['legal_type'] == 'legal entity'

    zh_br_json = ZH_Baurekurs(ZH_Baurekurs_test_string, namespace) 
    zh_br = json.loads(zh_br_json)
    assert zh_br['plaintiffs'][0]['legal_counsel'][0]['name'] == 'Dr. iur. Christof Truniger'
    assert zh_br['plaintiffs'][0]['legal_counsel'][0]['legal_type'] == 'natural person'
    assert zh_br['plaintiffs'][0]['legal_counsel'][0]['gender'] == 'male'

    zh_og_json = ZH_Obergericht(ZH_Obergericht_test_string, namespace)
    zh_og = json.loads(zh_og_json)
    assert zh_og['plaintiffs'][0]['legal_counsel'][0]['name'] == 'Dr. iur. X1._'
    assert zh_og['plaintiffs'][0]['legal_counsel'][0]['legal_type'] == 'natural person'
    assert zh_og['plaintiffs'][0]['legal_counsel'][0]['gender'] == 'female'
    assert zh_og['plaintiffs'][0]['legal_counsel'][1]['name'] == 'Dr. iur. X2._'
    assert zh_og['plaintiffs'][0]['legal_counsel'][1]['legal_type'] == 'natural person'
    assert zh_og['plaintiffs'][0]['legal_counsel'][1]['gender'] == 'male'

    zh_vg_json = ZH_Verwaltungsgericht(ZH_Verwaltungsgericht_test_string, namespace)
    zh_vg = json.loads(zh_vg_json)
    assert zh_vg['plaintiffs'][0]['legal_counsel'][0]['name'] == 'B'
    assert zh_vg['plaintiffs'][0]['legal_counsel'][0]['legal_type'] == 'natural person'
    assert zh_vg['plaintiffs'][0]['legal_counsel'][0]['gender'] == 'unknown'

    zh_svg_json = ZH_Sozialversicherungsgericht(ZH_Sozialversicherungsgericht_test_string, namespace)
    zh_svg = json.loads(zh_svg_json)
    assert zh_svg['plaintiffs'][0]['legal_counsel'][0]['name'] == 'Dr. Kreso Glavas'
    assert zh_svg['plaintiffs'][0]['legal_counsel'][0]['legal_type'] == 'natural person'
    assert zh_svg['plaintiffs'][0]['legal_counsel'][0]['gender'] == 'male'


# uncomment to test
# testing()