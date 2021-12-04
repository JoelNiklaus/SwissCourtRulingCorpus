from pathlib import Path
import re
import json
from typing import Dict, Optional, Tuple
from scrc.data_classes.court_composition import CourtComposition
from scrc.data_classes.court_person import CourtPerson

from scrc.enums.court_role import CourtRole
from scrc.enums.gender import Gender
from scrc.enums.language import Language
from scrc.enums.political_party import PoliticalParty
from scrc.enums.section import Section

"""
This file is used to extract the judicial persons from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
Overview of spiders still todo: https://docs.google.com/spreadsheets/d/1FZmeUEW8in4iDxiIgixY4g0_Bbg342w-twqtiIu8eZo/edit#gid=0
"""


def XX_SPIDER(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    # This is an example spider. Just copy this method and adjust the method name and the code to add your new spider.

    # header = sections[Section.HEADER] to get a specific section
    pass


# check if court got assigned shortcut: SELECT count(*) from de WHERE lower_court is not null and lower_court <> 'null' and lower_court::json#>>'{court}'~'[A-Z0-9_]{2,}';
def CH_BGer(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract judicial persons from decisions of the Federal Supreme Court of Switzerland
    :param sections:    the dict containing the sections per section key
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict
    """

    header = sections[Section.HEADER]

    information_start_regex = r'Besetzung|Bundesrichter|Composition( de la Cour:)?|Composizione|Giudic[ie] federal|composta'
    role_regexes = {
        Gender.MALE: {
            CourtRole.JUDGE: [r'Bundesrichter(?!in)', r'MM?\.(( et|,) Mmes?)? les? Juges?( fédéra(l|ux))?',
                       r'[Gg]iudici federali'],
            CourtRole.CLERK: [r'Gerichtsschreiber(?!in)', r'Greffier[^\w\s]*', r'[Cc]ancelliere']
        },
        Gender.FEMALE: {
            CourtRole.JUDGE: [r'Bundesrichterin(nen)?', r'Mmes? l(a|es) Juges? (fédérales?)?',
                       r'MMe et MM?\. les? Juges?( fédéra(l|ux))?', r'[Gg]iudice federal'],
            CourtRole.CLERK: [r'Gerichtsschreiberin(nen)?', r'Greffière.*Mme', r'[Cc]ancelliera']
        }
    }

    skip_strings = get_skip_strings()

    start_pos = re.search(information_start_regex, header)
    if start_pos:
        header = header[start_pos.span()[0]:]
    end_pos = {
        Language.DE: re.search(r'.(?=(1.)?(Partei)|(Verfahrensbeteiligt))', header) or re.search('Urteil vom',
                                                                                          header) or re.search(
            r'Gerichtsschreiber(in)?\s\w*.', header) or re.search(r'[Ii]n Sachen', header) or re.search(r'\w{2,}\.',
                                                                                                        header),
        Language.FR: re.search(r'.(?=(Parties|Participant))', header) or re.search(r'Greffi[eè]re? M(\w)*\.\s\w*.', header),
        Language.IT: re.search(r'.(?=(Parti)|(Partecipant))', header) or re.search(r'[Cc]ancellier[ae]:?\s\w*.',
                                                                            header) or re.search(r'\w{2,}\.', header),
    }
    end_pos = end_pos[namespace['language']]
    if end_pos:
        header = header[:end_pos.span()[1] - 1]

    header = header.replace(';', ',')
    header = header.replace('Th', '')
    header = header.replace(' und ', ', ')
    header = header.replace(' et ', ', ')
    header = header.replace(' e ', ', ')
    header = header.replace('MMe', 'Mme')
    header = re.sub(r'(?<!M)(?<!Mme)(?<!MM)(?<!\s\w)\.', ', ', header)
    header = re.sub(r'MM?\., Mme', 'M. et Mme', header)
    header = re.sub(r'Mmes?, MM?\.', 'MMe et M', header)
    header = header.replace('federali, ', 'federali')
    besetzungs_strings = header.split(',')

    def prepare_french_name_and_find_gender(person: CourtPerson) -> CourtPerson:
        """Removes the prefix from a french name and sets gender"""
        if person.name.find('M. ') > -1:
            person.name = person.name.replace('M. ', '')
            person.gender = Gender.MALE
        elif person.name.find('Mme') > -1:
            person.name = person.name.replace('Mme ', '')
            person.gender = Gender.FEMALE
        return CourtPerson

    besetzung = CourtComposition()
    current_role = CourtRole.JUDGE
    last_person: CourtPerson = None
    last_gender = Gender.MALE

    for text in besetzungs_strings:
        text = text.strip()
        if len(text) == 0 or text in skip_strings[namespace['language']]:
            continue
        if re.search(r'(?<![Vv]ice-)[Pp]r[äée]sid',
                     text):  # Set president either to the current person or the last Person (case 1: Präsident Niklaus, case 2: Niklaus, Präsident)
            if last_person:
                besetzung.president = last_person
                continue
            else:
                text = text.split()[-1]
                president, _ = match_person_to_database(CourtPerson(text), last_gender)
                besetzung.president = president
        has_role_in_string = False
        matched_gender_regex = False
        for gender in role_regexes:  # check for male and female all roles
            if matched_gender_regex:
                break
            role_regex = role_regexes[gender]
            for regex_key in role_regex:  # check each role
                regex = '|'.join(role_regex[regex_key])
                role_pos = re.search(regex, text)
                if role_pos: # Found a role regex
                    last_role = current_role
                    current_role = regex_key
                    name_match = re.search(r'[A-Z][A-Za-z\-éèäöü\s]*(?= Urteil)|[A-Z][A-Za-z\-éèäöü\s]*(?= )',
                                           text[role_pos.span()[1] + 1:])
                    name = name_match.group() if name_match else text[role_pos.span()[1] + 1:]
                    if len(name.strip()) == 0:
                        if (last_role == CourtRole.CLERK and len(besetzung.clerks) == 0) or (last_role == CourtRole.JUDGE and len(besetzung.judges) == 0):
                            break

                        last_person_name = besetzung.clerks.pop().name if (last_role == CourtRole.CLERK) else besetzung.clerks.pop().name # rematch in database with new role
                        last_person_new_match, _ = match_person_to_database(CourtPerson(name=last_person_name, court_role=current_role), gender)
                        if current_role == CourtRole.JUDGE:
                            besetzung.judges.append(last_person_new_match)
                        elif current_role == CourtRole.CLERK:
                            besetzung.clerks.append(last_person_new_match)
                    if namespace['language'] == Language.FR:
                        person = prepare_french_name_and_find_gender(name)
                        gender = person.gender or gender
                        person.court_role = current_role
                    matched_person, _ = match_person_to_database(person, gender)
                    if current_role == CourtRole.JUDGE:
                        besetzung.judges.append(matched_person)
                    elif current_role == CourtRole.CLERK:
                        besetzung.clerks.append(matched_person)
                    last_person = matched_person
                    last_gender = matched_person.gender
                    has_role_in_string = True
                    matched_gender_regex = True
                    break
        if not has_role_in_string:  # Current string has no role regex match
            if current_role not in besetzung:
                besetzung[current_role] = []
            if namespace['language'] == Language.FR:
                person = prepare_french_name_and_find_gender(text)
                last_gender = person.gender or last_gender
            name_match = re.search(
                r'[A-Z][A-Za-z\-éèäöü\s]*(?= Urteil)|[A-Z][A-Za-z\-éèäöü\s]*(?= )|[A-Z][A-Za-z\-éèäöü\s]*', person.name)
            if not name_match:
                continue
            name = name_match.group()
            person.court_role = current_role
            matched_person, _ = match_person_to_database(person, last_gender)
            if current_role == CourtRole.JUDGE:
                besetzung.judges.append(matched_person)
            elif current_role == CourtRole.CLERK:
                besetzung.clerks.append(matched_person)
            last_person = person
    return besetzung


def ZG_Verwaltungsgericht(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract judicial persons from decisions of the Verwaltungsgericht of Zug
    :param header:      the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict
    """
    
    header = sections[Section.HEADER]

    role_regexes = {
        Gender.MALE: {
            CourtRole.JUDGE: [r'Richter(?!in)', r'Einzelrichter(?!in)'],
            CourtRole.CLERK: [r'Gerichtsschreiber(?!in)']
        },
        Gender.FEMALE: {
            CourtRole.JUDGE: [r'Richterin(nen)?',r'Einzelrichterin(nen)?'],
            CourtRole.CLERK: [r'Gerichtsschreiberin(nen)?']
        }
    }

    header = header.replace('U R T E I L', 'Urteil')
    header = header.replace('URTEIL', 'Urteil')

    information_start_regex = r'Mitwirkende|Einzelrichter'
    start_pos = re.search(information_start_regex, header)
    if start_pos:
        # split off the first word
        header = header[start_pos.span()[1]:]

    information_end_regex = r'Urteil'
    end_pos = re.search(information_end_regex, header)
    if end_pos:
        header = header[:end_pos.span()[1]]
        # split off the last word
        header = header.rsplit(' ', 1)[0]

    # this puts a comma before every judicial role
    for gender in role_regexes:
        for regex_key in role_regexes[gender]:
            for regex in role_regexes[gender][regex_key]:
                regex_group = r"(" + regex + r")"
                header = re.sub(regex_group, r', \1', header)

    besetzung = CourtComposition()
    besetzung = find_besetzung(header, role_regexes, namespace)
    return besetzung


def ZH_Baurekurs(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract judicial persons from decisions of the Baurekursgericht of Zurich
    :param header:      the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict
    """
    header = sections[Section.HEADER]
    print(header)

    role_regexes = {
        Gender.MALE: {
            CourtRole.JUDGE: [r'Abteilungspräsident(?!in)', r'Baurichter(?!in)', r'Abteilungsvizepräsident(?!in)', r'Ersatzrichter(?!in)'],
            CourtRole.CLERK: [r'Gerichtsschreiber(?!in)']
        },
        Gender.FEMALE: {
            CourtRole.JUDGE: [r'Abteilungspräsidentin(nen)?',r'Baurichterin(nen)?', r'Abteilungsvizepräsidentin(nen)?', r'Ersatzrichterin(nen)?'],
            CourtRole.CLERK: [r'Gerichtsschreiberin(nen)?']
        }
    }

    information_start_regex = r'Mitwirkende'
    start_pos = re.search(information_start_regex, header)
    if start_pos:
        # split off the first word
        header = header[start_pos.span()[1]:]


    information_end_regex = r'in Sachen'
    end_pos = re.search(information_end_regex, header)
    if end_pos:
        header = header[:end_pos.span()[1]]
        # split off the last two words
        header = header.rsplit(' ', 2)[0]

    besetzung = CourtComposition()
    besetzung = find_besetzung(header, role_regexes, namespace)
    return besetzung


def ZH_Obergericht(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract judicial persons from decisions of the Obergericht of Zurich
    :param header:      the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict
    """

    header = sections[Section.HEADER]

    role_regexes = {
        Gender.MALE: {
            CourtRole.JUDGE: [r'Oberrichter(?!in)', r'Ersatzrichter(?!in)', r'Kassationsrichter(?!in)', r'Vizepräsident(?!in)', r'Bezirksrichter(?!in)'],
            CourtRole.CLERK: [r'Gerichtsschreiber(?!in)', r'Sekretär(?!in)']
        },
        Gender.FEMALE: {
            CourtRole.JUDGE: [r'Oberrichterin(nen)?', r'Ersatzrichterin(nen)?', r'Kassationsrichterin(nen)?', r'Vizepräsidentin(nen)?', r'Bezirksrichterin(nen)?'],
            CourtRole.CLERK: [r'Gerichtsschreiberin(nen)?', r'Sekretärin(nen)?']
        }
    }

    information_start_regex = r'Mitwirkend'
    start_pos = re.search(information_start_regex, header)
    if start_pos:
        # split off the first word
        header = header[start_pos.span()[1]:]
    
    information_end_regex = r'Zirkulationsbeschluss vom|Beschluss vom|Urteil vom|Beschluss und Urteil vom|in Sachen'
    end_pos = re.search(information_end_regex, header)
    if end_pos:
        header = header[:end_pos.span()[1]]
        # splitt off the last two words
        header = header.rsplit(' ', 2)[0]

    besetzung = CourtComposition()
    besetzung = find_besetzung(header, role_regexes, namespace)
    return besetzung



def ZH_Sozialversicherungsgericht(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract judicial persons from decisions of the Sozialversicherungsgericht of Zurich
    :param header:      the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict
    """

    header = sections[Section.HEADER]

    role_regexes = {
        Gender.MALE: {
            CourtRole.JUDGE: [r'Sozialversicherungsrichter(?!in)', r'Ersatzrichter(?!in)'],
            CourtRole.CLERK: [r'Gerichtsschreiber(?!in)']
        },
        Gender.FEMALE: {
            CourtRole.JUDGE: [r'Sozialversicherungsrichterin(nen)?',r'Ersatzrichterin(nen)?'],
            CourtRole.CLERK: [r'Gerichtsschreiberin(nen)?']
        }
    }

    information_start_regex = r'Mitwirkende|Einzelrichter|Kammer'
    start_pos = re.search(information_start_regex, header)
    if start_pos:
        # split off the first word
        header = header[start_pos.span()[1]:]

    information_end_regex = r'Urteil vom|in Sachen'
    end_pos = re.search(information_end_regex, header)
    if end_pos:
        header = header[:end_pos.span()[1]]
        # split off the last two words
        header = header.rsplit(' ', 2)[0]

    # this puts a comma before every judicial role
    for gender in role_regexes:
        for regex_key in role_regexes[gender]:
            for regex in role_regexes[gender][regex_key]:
                regex_group = r"(" + regex + r")"
                header = re.sub(regex_group, r', \1', header)

    besetzung = CourtComposition()
    besetzung = find_besetzung(header, role_regexes, namespace)
    return besetzung


def ZH_Steuerrekurs(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract judicial persons from decisions of the Steuerrekursgericht of Zurich
    :param header:      the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict
    """

    header = sections[Section.HEADER]

    role_regexes = {
        Gender.MALE: {
            CourtRole.JUDGE: [r'Abteilungspräsident(?!in)', r'Steuerrichter(?!in)', r'Ersatzrichter(?!in)', r'Einzelrichter(?!in)'],
            CourtRole.CLERK: [r'Gerichtsschreiber(?!in)']
        },
        Gender.FEMALE: {
            CourtRole.JUDGE: [r'Abteilungspräsidentin(nen)?',r'Steuerrichterin(nen)?',r'Ersatzrichterin(nen)?',r'Einzelrichterin(nen)?'],
            CourtRole.CLERK: [r'Gerichtsschreiberin(nen)?']
        }
    }

    information_start_regex = r'Mitwirkend'
    start_pos = re.search(information_start_regex, header)
    if start_pos:
        # split off the first word
        header = header[start_pos.span()[1]:]
    
    information_end_regex = r'In Sachen|in Sachen'
    end_pos = re.search(information_end_regex, header)
    if end_pos:
        header = header[:end_pos.span()[1]]
        # split off the first word
        header = header.rsplit(' ', 2)[0]

    besetzung = CourtComposition()
    besetzung = find_besetzung(header, role_regexes, namespace)
    return besetzung


def ZH_Verwaltungsgericht(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract judicial persons from decisions of the Verwaltungsgericht of Zurich
    :param header:      the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict
    """

    header = sections[Section.HEADER]

    role_regexes = {
        Gender.MALE: {
            CourtRole.JUDGE: [r'Abteilungspräsident(?!in)', r'Verwaltungsrichter(?!in)'],
            CourtRole.CLERK: [r'Gerichtsschreiber(?!in)']
        },
        Gender.FEMALE: {
            CourtRole.JUDGE: [r'Abteilungspräsidentin(nen)?',r'Verwaltungsrichterin(nen)?'],
            CourtRole.CLERK: [r'Gerichtsschreiberin(nen)?']
        }
    }

    information_start_regex = r'Mitwirkend'
    start_pos = re.search(information_start_regex, header)
    if start_pos:
        # split off the first word
        header = header[start_pos.span()[1]:]

    information_end_regex = r'In Sachen|in Sachen'
    end_pos = re.search(information_end_regex, header)
    if end_pos:
        header = header[:end_pos.span()[1]]
        # split off the last two words
        header = header.rsplit(' ', 2)[0]

    besetzung = CourtComposition()
    besetzung = find_besetzung(header, role_regexes, namespace)
    return besetzung



def get_besetzungs_strings(header: str) -> list:
    """
    Modifies the header of a decision and turns it into a list
    :header:    the header of a decision
    :return:    a list of besetzungs_strings
    """
    # regularize different forms to denote the Vorsitz
    header = header.replace('(Vorsitz)', 'Vorsitz')
    header = header.replace('Vorsitzender', 'Vorsitz')
    header = header.replace('Vorsitzende', 'Vorsitz')
    header = header.replace('Vorsitz', ', Vorsitz, ')
    # these word separators aren't relevant here
    header = header.replace('- ', '')
    header = header.replace(':', '')
    # a semicolon can be treated as a comma here
    header = header.replace(';', ',')
    # der & die aren't relevant for this task
    header = header.replace(' der ', ' ')
    header = header.replace(' die ', ' ')
    header = header.replace('  ', ' ')
    # und & sowie separte different people
    header = header.replace(' und', ', ')
    header = header.replace(' sowie', ', ')
    # academic degrees aren't relevant for this task
    header = header.replace('lic. ', '')
    header = header.replace('iur. ', '')
    header = header.replace('Dr. ', '')
    header = header.replace('MLaw ', '')
    header = header.replace('M.A. ', '')
    header = header.replace('HSG ', '')
    header = header.replace('PD ', '')
    # neither is this relevant
    header = header.replace(' als Einzelrichterin', '')
    header = header.replace(' als Einzelrichter', '')
    # uncomment to debug
    # print(header)
    return header.split(',')


def get_skip_strings() -> dict:
    """
    :return:    strings which should be skipped when extracting the court composition
    """
    return {
        Language.DE: ['Einzelrichter', 'Konkurskammer', 'Beschwerdeführerin', 'Beschwerdeführer', 'Kläger', 'Berufungskläger'],
        Language.FR: ['Juge suppléant', 'en qualité de juge unique'],
        Language.IT: ['Giudice supplente', 'supplente']
    }


def match_person_to_database(person: CourtPerson, current_gender: Gender) -> Tuple[CourtPerson, bool]:
    """"Matches a name of a given role to a person from personal_information.json"""
    personal_information_database = json.loads(Path("personal_information.json").read_text())

    results = []
    name = person.name.replace('.', '').strip()
    split_name = name.split()
    initial = False
    if len(split_name) > 1:
        initial = next((x for x in split_name if len(x) == 1), None)
        split_name = list(filter(lambda x: len(x) > 1, split_name))
    if person.court_role.value in personal_information_database:
        for subcategory in personal_information_database[person.court_role]:
            for cat_id in personal_information_database[person.court_role][subcategory]:
                for db_person in personal_information_database[person.court_role][subcategory][cat_id]:
                    if set(split_name).issubset(set(db_person['name'].split())):
                        if not initial or re.search(rf'\s{initial.upper()}\w*', db_person['name']):
                            person.name = db_person['name']
                            if db_person.get('gender'):
                                person.gender = Gender(db_person['gender'])
                            if db_person.get('party'):
                                person.party = PoliticalParty(db_person['party'])
                            results.append(person)
    else:
        for existing_role in personal_information_database:
            temp_person = CourtPerson(person.name, court_role=CourtRole(existing_role))
            db_person, match = match_person_to_database(temp_person, current_gender)
            if match:
                results.append(db_person)
    if len(results) == 1:
        if not results[0].gender:
            results[0].gender = current_gender
        return person, True
    return person, False


def find_besetzung(header: str, role_regexes: dict, namespace: dict) -> CourtComposition:
    """
    Find the court composition in the header of a decision
    :param header:          the string containing the header
    :param role_regexes:    the regexes for the court person roles
    :param namespace:       the namespace containing some metadata of the court decision
    :return:                the court composition
    """
    skip_strings = get_skip_strings()
    besetzungs_strings = get_besetzungs_strings(header)
    besetzung = CourtComposition()
    current_role = CourtRole.JUDGE
    last_person: CourtPerson = None
    person: CourtPerson = None
    last_gender = Gender.UNKNOWN

    for text in besetzungs_strings:
        text = text.strip()
        # delete the last character if it's a dot following a lower-case character
        if re.search(r'[a-z]\.$', text):
            text=text[:-1]
        if len(text) == 0 or text in skip_strings[namespace['language']]:
            continue
        if (re.search(r'Vorsitz', text) or re.search(r'(?<![Vv]ize)[Pp]räsident', text)):  
        # Set president either to the current person or the last Person (case 1: Präsident Niklaus, case 2: Niklaus, Präsident)
            if last_person:
                besetzung.president = last_person
                continue
            else:
                pos = re.search(r'(?<![Vv]ize)[Pp]räsident(in)?', text)
                # assign gender depending on the noun ending
                if re.search(r'räsidentin', text):
                    last_gender = Gender.FEMALE
                elif re.search(r'räsident\b', text):
                    last_gender = Gender.MALE
                text = text[pos.span()[1]:]
                text = text.strip()
                besetzung.president = CourtPerson(text, last_gender)
        has_role_in_string = False
        matched_gender_regex = False


        for gender in role_regexes:  # check for male and female all roles
            if matched_gender_regex:
                break
            role_regex = role_regexes[gender]
            for regex_key in role_regex:  # check each role
                regex = '|'.join(role_regex[regex_key])
                role_pos = re.search(regex, text)
                if role_pos: # Found a role regex
                    last_role = current_role
                    current_role = regex_key
                    # a name can consist of letters, periods, dashes, and spaces
                    name_match = re.search(r'[A-Za-zÀ-ž\.\- ]+', text[role_pos.span()[1] + 1:])

                    name = name_match.group() if name_match else text[role_pos.span()[1] + 1:]
                    if len(name.strip()) == 0:
                        if (last_role == CourtRole.CLERK and len(besetzung.clerks) == 0) or (last_role == CourtRole.JUDGE and len(besetzung.judges) == 0):
                            break

                        last_person_name = besetzung.clerks.pop().name if (last_role == CourtRole.CLERK) else besetzung.clerks.pop().name # rematch in database with new role
                        last_person_new_match = CourtPerson(last_person_name, gender, current_role)
                        if current_role == CourtRole.JUDGE:
                            besetzung.judges.append(last_person_new_match)
                        elif current_role == CourtRole.CLERK:
                            besetzung.clerks.append(last_person_new_match)
                    matched_person = CourtPerson(name, gender, current_role)
                    if current_role == CourtRole.JUDGE:
                        besetzung.judges.append(matched_person)
                    elif current_role == CourtRole.CLERK:
                        besetzung.clerks.append(matched_person)
                    last_person = matched_person
                    last_gender = matched_person.gender
                    has_role_in_string = True
                    matched_gender_regex = True
                    break
        if not has_role_in_string:  # Current string has no role regex match
            # a name can consist of letters, periods, dashes, and spaces
            name_match = re.search(r'[A-Za-zÀ-ž\.\- ]+', text)
            if not name_match:
                continue
            name = name_match.group()
            person = CourtPerson(name, last_gender, current_role)
            matched_person = person
            if current_role == CourtRole.JUDGE:
                besetzung.judges.append(matched_person)
            elif current_role == CourtRole.CLERK:
                besetzung.clerks.append(matched_person)
            last_person = person
    # uncomment to debug
    # print(besetzung)
    return besetzung




def testing():
    """
    This function tests the extracting functions checking if their output is correct given a test header (that was copied from the section splitting output)
    """
    ZG_Verwaltungsgericht_test_header = ['Normal.dot', 'VERWALTUNGSGERICHT DES KANTONS ZUG', 'SOZIALVERSICHERUNGSRECHTLICHE KAMMER', 'Mitwirkende Richter: lic. iur. Adrian Willimann, Vorsitz lic. iur. Jacqueline Iten-Staub und Dr. iur. Matthias Suter Gerichtsschreiber: MLaw Patrick Trütsch', 'U R T E I L vom 18. Juni 2020 [rechtskräftig] gemäss § 29 der Geschäftsordnung', 'in Sachen', 'A._ Beschwerdeführer vertreten durch B._ AG', 'gegen', 'Ausgleichskasse Zug, Baarerstrasse 11, Postfach, 6302 Zug Beschwerdegegnerin', 'betreffend', 'Ergänzungsleistungen (hypothetisches Erwerbseinkommen)', 'S 2019 121', '2', 'Urteil S 2019 121']

    ZH_Steuerrekurs_test_header = ['Endentscheid Kammer', 'Steuerrekursgericht des Kantons Zürich', '2. Abteilung', '2 DB.2017.240 2 ST.2017.296', 'Entscheid', '5. Februar 2019', 'Mitwirkend:', 'Abteilungspräsident Christian Mäder, Steuerrichterin Micheline Roth, Steuerrichterin Barbara Collet und Gerichtsschreiber Hans Heinrich Knüsli', 'In Sachen', '1. A, 2. B,', 'Beschwerdeführer/ Rekurrenten, vertreten durch C AG,', 'gegen', '1. Schw eizer ische E idgenossenschaf t , Beschwerdegegnerin, 2. Staat Zür ich , Rekursgegner, vertreten durch das kant. Steueramt, Division Konsum, Bändliweg 21, Postfach, 8090 Zürich,', 'betreffend', 'Direkte Bundessteuer 2012 sowie Staats- und Gemeindesteuern 2012', '- 2 -', '2 DB.2017.240 2 ST.2017.296']

    ZH_Baurekurs_test_header = ['BRGE Nr. 0/; GUTH vom', 'Baurekursgericht des Kantons Zürich', '2. Abteilung', 'G.-Nr. R2.2018.00197 und R2.2019.00057 BRGE II Nr. 0142/2019 und 0143/2019', 'Entscheid vom 10. September 2019', 'Mitwirkende Abteilungsvizepräsident Adrian Bergmann, Baurichter Stefano Terzi,  Marlen Patt, Gerichtsschreiber Daniel Schweikert', 'in Sachen Rekurrentin', 'V. L. [...]', 'vertreten durch [...]', 'gegen Rekursgegnerschaft', '1. Baubehörde X 2. M. I. und K. I.-L. [...]', 'Nr. 2 vertreten durch [...]', 'R2.2018.00197 betreffend Baubehördenbeschluss vom 4. September 2017; Baubewilligung für Um-', 'bau Einfamilienhausteil und Ausbau Dachgeschoss, [...], BRGE II Nr. 00025/2018 vom 6. März 2018; Rückweisung zum  mit VB.2018.00209 vom 20. September 2018', 'R2.2019.00057 Präsidialverfügung vom 29. März 2019; Baubewilligung für Umbau  und Ausbau Dachgeschoss (1. Projektänderung), [...] _', 'R2.2018.00197 Seite 2']

    ZH_Obergericht_test_header = ['Urteil - Abweisung, begründet', 'Bezirksgericht Zürich 3. Abteilung', 'Geschäfts-Nr.: CG170019-L / U', 'Mitwirkend: Vizepräsident lic. iur. Th. Kläusli, Bezirksrichter lic. iur. K. Vogel,', 'Ersatzrichter MLaw D. Brugger sowie der Gerichtsschreiber M.A.', 'HSG Ch. Reitze', 'Urteil vom 4. März 2020', 'in Sachen', 'A._, Kläger', 'vertreten durch Rechtsanwalt lic. iur. W._', 'gegen', '1. B._, 2. C._-Stiftung, 3. D._, Beklagte', '1 vertreten durch Rechtsanwalt Dr. iur. X._', '2 vertreten durch Rechtsanwältin Dr. iur. Y._']

    ZH_Verwaltungsgericht_test_header = ['Verwaltungsgericht des Kantons Zürich 4. Abteilung', 'VB.2020.00452', 'Urteil', 'der 4. Kammer', 'vom 24. September 2020', 'Mitwirkend: Abteilungspräsidentin Tamara Nüssle (Vorsitz), Verwaltungsrichter Reto Häggi Furrer, Verwaltungsrichter Martin Bertschi, Gerichtsschreiber David Henseler.', 'In Sachen', 'A, vertreten durch RA B,', 'Beschwerdeführerin,', 'gegen', 'Migrationsamt des Kantons Zürich,', 'Beschwerdegegner,', 'betreffend vorzeitige Erteilung der Niederlassungsbewilligung,']

    ZH_Sozialversicherungsgericht_test_header = ['Sozialversicherungsgerichtdes Kantons ZürichIV.2014.00602', 'II. Kammer', 'Sozialversicherungsrichter Mosimann, Vorsitzender', 'Sozialversicherungsrichterin Käch', 'Sozialversicherungsrichterin Sager', 'Gerichtsschreiberin Kudelski', 'Urteil vom 11. August 2015', 'in Sachen', 'X._', 'Beschwerdeführerin', 'vertreten durch Rechtsanwalt Dr. Kreso Glavas', 'Advokatur Glavas AG', 'Markusstrasse 10, 8006 Zürich', 'gegen', 'Sozialversicherungsanstalt des Kantons Zürich, IV-Stelle', 'Röntgenstrasse 17, Postfach, 8087 Zürich', 'Beschwerdegegnerin', 'weitere Verfahrensbeteiligte:', 'Personalvorsorgestiftung der Y._', 'Beigeladene']


    ZG_Verwaltungsgericht_test_string = ' '.join(map(str, ZG_Verwaltungsgericht_test_header))
    ZH_Steuerrekurs_test_string = ' '.join(map(str, ZH_Steuerrekurs_test_header))
    ZH_Baurekurs_test_string = ' '.join(map(str, ZH_Baurekurs_test_header))
    ZH_Obergericht_test_string = ' '.join(map(str, ZH_Obergericht_test_header))
    ZH_Verwaltungsgericht_test_string = ' '.join(map(str, ZH_Verwaltungsgericht_test_header))
    ZH_Sozialversicherungsgericht_test_string = ' '.join(map(str, ZH_Sozialversicherungsgericht_test_header))

    namespace = {'language' : Language.DE}

    zg_vg = ZG_Verwaltungsgericht(ZG_Verwaltungsgericht_test_string, namespace)
    # No tests for the gender because this court uses a generic masculine noun for multiple judges
    assert zg_vg.president.name == 'Adrian Willimann'
    assert zg_vg.judges[0].name == 'Adrian Willimann'
    assert zg_vg.judges[1].name == 'Jacqueline Iten-Staub'
    assert zg_vg.judges[2].name == 'Matthias Suter'
    assert zg_vg.clerks[0].name == 'Patrick Trütsch'
    zh_sr = ZH_Steuerrekurs(ZH_Steuerrekurs_test_string, namespace)
    assert zh_sr.president.name == 'Christian Mäder'
    assert zh_sr.president.gender.value == 'male'
    assert zh_sr.judges[0].name == 'Christian Mäder'
    assert zh_sr.judges[0].gender.value == 'male'
    assert zh_sr.judges[1].name == 'Micheline Roth'
    assert zh_sr.judges[1].gender.value == 'female'
    assert zh_sr.judges[2].name == 'Barbara Collet'
    assert zh_sr.judges[2].gender.value == 'female'
    assert zh_sr.clerks[0].name == 'Hans Heinrich Knüsli'
    assert zh_sr.clerks[0].gender.value == 'male'
    zh_br = ZH_Baurekurs(ZH_Baurekurs_test_string, namespace) 
    assert zh_br.president == None
    assert zh_br.judges[0].name == 'Adrian Bergmann'
    assert zh_br.judges[0].gender.value == 'male'
    assert zh_br.judges[1].name == 'Stefano Terzi'
    assert zh_br.judges[1].gender.value == 'male'
    assert zh_br.judges[2].name == 'Marlen Patt'
    assert zh_br.judges[2].gender.value == 'male'
    assert zh_br.clerks[0].name == 'Daniel Schweikert'
    assert zh_br.clerks[0].gender.value == 'male'
    zh_og = ZH_Obergericht(ZH_Obergericht_test_string, namespace)
    assert zh_og.president == None
    assert zh_og.judges[0].name == 'Th. Kläusli'
    assert zh_og.judges[0].gender.value == 'male'
    assert zh_og.judges[1].name == 'K. Vogel'
    assert zh_og.judges[1].gender.value == 'male'
    assert zh_og.judges[2].name == 'D. Brugger'
    assert zh_og.judges[2].gender.value == 'male'
    assert zh_og.clerks[0].name == 'Ch. Reitze'
    assert zh_og.clerks[0].gender.value == 'male'
    zh_vg= ZH_Verwaltungsgericht(ZH_Verwaltungsgericht_test_string, namespace)
    assert zh_vg.president.name == 'Tamara Nüssle'
    assert zh_vg.president.gender.value == 'female'
    assert zh_vg.judges[0].name == 'Tamara Nüssle'
    assert zh_vg.judges[0].gender.value == 'female'
    assert zh_vg.judges[1].name == 'Reto Häggi Furrer'
    assert zh_vg.judges[1].gender.value == 'male'
    assert zh_vg.judges[2].name == 'Martin Bertschi'
    assert zh_vg.judges[2].gender.value == 'male'
    assert zh_vg.clerks[0].name == 'David Henseler'
    assert zh_vg.clerks[0].gender.value == 'male'
    zh_svg = ZH_Sozialversicherungsgericht(ZH_Sozialversicherungsgericht_test_string, namespace)
    assert zh_svg.president.name == 'Mosimann'
    assert zh_svg.president.gender.value == 'male'
    assert zh_svg.judges[0].name == 'Mosimann'
    assert zh_svg.judges[0].gender.value == 'male'
    assert zh_svg.judges[1].name == 'Käch'
    assert zh_svg.judges[1].gender.value == 'female'
    assert zh_svg.judges[2].name == 'Sager'
    assert zh_svg.judges[2].gender.value == 'female'
    assert zh_svg.clerks[0].name == 'Kudelski'
    assert zh_svg.clerks[0].gender.value == 'female'

# uncomment to test
# testing()
