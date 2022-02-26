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
    Extract the court composition from decisions of the Verwaltungsgericht of Zug
    :param header:      the dict containing the sections per section key
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the court composition of a decision
    """
    
    header = sections[Section.HEADER]

    role_regexes = {
        Gender.MALE: {
            CourtRole.JUDGE: [r'Richter(?!in)', r'Einzelrichter(?!in)', r'Schiedsrichter(?!in)'],
            CourtRole.CLERK: [r'Gerichtsschreiber(?!in)']
        },
        Gender.FEMALE: {
            CourtRole.JUDGE: [r'Richterin(nen)?', r'Einzelrichterin(nen)?', r'Schiedsrichterin(nen)?'],
            CourtRole.CLERK: [r'Gerichtsschreiberin(nen)?']
        }
    }

    # regularize different forms of words
    header = header.replace('U R T E I L', 'Urteil')
    header = header.replace('U R TE I L', 'Urteil')
    header = header.replace('URTEIL', 'Urteil')
    header = header.replace('Z W I S C H E N E N T S C H E I D', 'Zwischenentscheid')

    information_start_regex = r'Mitwirkende|Einzelrichter'
    start_pos = re.search(information_start_regex, header)
    if start_pos:
        # split off the first word
        header = header[start_pos.span()[1]:]

    information_end_regex = r'Urteil|Zwischenentscheid'
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

    composition = CourtComposition()
    composition = find_composition(header, role_regexes, namespace)
    return composition.toJSON() if composition else None


def ZH_Baurekurs(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract the court composition from decisions of the Baurekursgericht of Zurich
    :param header:      the dict containing the sections per section key
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the court composition
    """

    header = sections[Section.HEADER]

    role_regexes = {
        Gender.MALE: {
            CourtRole.JUDGE: [r'Abteilungspräsident(?!in)', r'Baurichter(?!in)', r'Abteilungsvizepräsident(?!in)', r'Ersatzrichter(?!in)'],
            CourtRole.CLERK: [r'Gerichtsschreiber(?!in)']
        },
        Gender.FEMALE: {
            CourtRole.JUDGE: [r'Abteilungspräsidentin(nen)?', r'Baurichterin(nen)?', r'Abteilungsvizepräsidentin(nen)?', r'Ersatzrichterin(nen)?'],
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

    composition = CourtComposition()
    composition = find_composition(header, role_regexes, namespace)
    return composition.toJSON() if composition else None

def ZH_Obergericht(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract the court composition from decisions of the Obergericht of Zurich
    :param header:      the dict containing the sections per section key
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the court composition
    """

    header = sections[Section.HEADER]

    role_regexes = {
        Gender.MALE: {
            CourtRole.JUDGE: [r'Oberrichter(?!in)', r'Ersatzoberrichter(?!in)', r'Ersatzrichter(?!in)', r'Kassationsrichter(?!in)', r'Vizepräsident(?!in)', r'Bezirksrichter(?!in)', r'Handelsrichter(?!in)', r'Einzelrichter(?!in)'],
            CourtRole.CLERK: [r'Gerichtsschreiber(?!in)', r'Sekretär(?!in)']
        },
        Gender.FEMALE: {
            CourtRole.JUDGE: [r'Oberrichterin(nen)?', r'Ersatzoberrichterin(nen)?', r'Ersatzrichterin(nen)?', r'Kassationsrichterin(nen)?', r'Vizepräsidentin(nen)?', r'Bezirksrichterin(nen)?', r'Handelsrichterin(nen)?', r'Einzelrichterin(nen)?'],
            CourtRole.CLERK: [r'Gerichtsschreiberin(nen)?', r'Sekretärin(nen)?']
        }
    }

    information_start_regex = r'Mitwirkend'
    start_pos = re.search(information_start_regex, header)
    if start_pos:
        # split off the first word
        header = header[start_pos.span()[1]:]
    
    information_end_regex = r'Zirkulationsbeschluss vom|Beschluss vom|Urteil vom|Verfügung vom|Beschluss und|in Sachen'
    end_pos = re.search(information_end_regex, header)
    if end_pos:
        header = header[:end_pos.span()[1]]
        # split off the last two words
        header = header.rsplit(' ', 2)[0]

    composition = CourtComposition()
    composition = find_composition(header, role_regexes, namespace)
    return composition.toJSON() if composition else None


def ZH_Sozialversicherungsgericht(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract the court composition from decisions of the Sozialversicherungsgericht of Zurich
    :param header:      the dict containing the sections per section key
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the court composition
    """

    header = sections[Section.HEADER]

    role_regexes = {
        Gender.MALE: {
            CourtRole.JUDGE: [r'Sozialversicherungsrichter(?!in)', r'Ersatzrichter(?!in)', r'Schiedsrichter(?!in)'],
            CourtRole.CLERK: [r'Gerichtsschreiber(?!in)', r'Gerichtssekretär(?!in)']
        },
        Gender.FEMALE: {
            CourtRole.JUDGE: [r'Sozialversicherungsrichterin(nen)?', r'Ersatzrichterin(nen)?', r'Schiedsrichterin(nen)?'],
            CourtRole.CLERK: [r'Gerichtsschreiberin(nen)?', r'Gerichtssekretärin(nen)?']
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

    composition = CourtComposition()
    composition = find_composition(header, role_regexes, namespace)
    return composition.toJSON() if composition else None

def ZH_Steuerrekurs(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract the court composition from decisions of the Steuerrekursgericht of Zurich
    :param header:      the dict containing the sections per section key
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the court composition
    """

    header = sections[Section.HEADER]

    role_regexes = {
        Gender.MALE: {
            CourtRole.JUDGE: [r'Abteilungspräsident(?!in)', r'Abteilungsvizepräsident(?!in)', r'Steuerrichter(?!in)', r'Ersatzrichter(?!in)', r'Einzelrichter(?!in)'],
            CourtRole.CLERK: [r'Gerichtsschreiber(?!in)', r'Sekretär(?!in)']
        },
        Gender.FEMALE: {
            CourtRole.JUDGE: [r'Abteilungspräsidentin(nen)?', r'Abteilungsvizepräsidentin(nen)?', r'Steuerrichterin(nen)?', r'Ersatzrichterin(nen)?', r'Einzelrichterin(nen)?'],
            CourtRole.CLERK: [r'Gerichtsschreiberin(nen)?', r'Sekretärin(nen)?']
        }
        ,
        Gender.UNKNOWN: {
            CourtRole.JUDGE: [r'Ersatzmitglied(er)?', r'Mitglied(er)?']
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

    composition = CourtComposition()
    composition = find_composition(header, role_regexes, namespace)
    return composition.toJSON() if composition else None

def ZH_Verwaltungsgericht(sections: Dict[Section, str], namespace: dict) -> Optional[str]:
    """
    Extract the court composition from decisions of the Verwaltungsgericht of Zurich
    :param header:      the dict containing the sections per section key
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the court composition
    """

    header = sections[Section.HEADER]

    role_regexes = {
        Gender.MALE: {
            CourtRole.JUDGE: [r'Abteilungspräsident(?!in)', r'Abteilungsvizepräsident(?!in)', r'Verwaltungsrichter(?!in)', r'Ersatzrichter(?!in)'],
            CourtRole.CLERK: [r'Gerichtsschreiber(?!in)', r'Gerichtssekretär(?!in)', r'Sekretär(?!in)']
        },
        Gender.FEMALE: {
            CourtRole.JUDGE: [r'Abteilungspräsidentin(nen)?', r'Abteilungsvizepräsidentin(nen)?', r'Verwaltungsrichterin(nen)?', r'Ersatzrichterin(nen)?'],
            CourtRole.CLERK: [r'Gerichtsschreiberin(nen)?', r'Gerichtssekretärin(nen)?', r'Sekretärin(nen)?']
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

    composition = CourtComposition()
    composition = find_composition(header, role_regexes, namespace)
    return composition.toJSON() if composition else None



def get_composition_strings(header: str) -> list:
    """
    Modifies the header of a decision and turns it into a list
    :param header:  the header of a decision
    :return:        a list of composition_strings
    """
    # repeating commas aren't necessary
    header = re.sub(r', *, *', ', ', header)
    # trying to join words that are split over two lines
    header = re.sub(r'- *, *', '', header)
    header = header.replace('- ', '')
    # regularize different forms to denote the Vorsitz
    header = header.replace('(Vorsitz)', 'Vorsitz')
    header = header.replace('Vorsitzender', 'Vorsitz')
    header = header.replace('Vorsitzende', 'Vorsitz')
    header = header.replace('Vorsitz', ', Vorsitz, ')
    # these word separators aren't relevant here
    header = header.replace(':', '')
    # a semicolon can be treated as a comma here
    header = header.replace(';', ',')
    # der & die aren't relevant for this task
    header = re.sub(r'\bder\b', '', header)
    header = re.sub(r'\bdie\b', '', header)
    # und & sowie separte different people
    header = header.replace(' und', ', ')
    header = header.replace(' sowie', ', ')
    # academic degrees presumably aren't relevant for this task
    header = header.replace('lic.', '')
    header = header.replace('iur.', '')
    header = header.replace('Dr.', '')
    header = header.replace('Prof.', '')
    header = header.replace('MLaw ', '')
    header = header.replace('M.A.', '')
    header = header.replace('HSG ', '')
    header = header.replace('PD ', '')
    header = header.replace('a.o.', '')
    header = header.replace('LL.M.', '')
    header = header.replace('LL. M.', '')
    header = header.replace('LLM ', '')
    # delete multiple spaces
    header = header.strip()
    header = re.sub(' +', ' ', header)
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


def find_composition(header: str, role_regexes: dict, namespace: dict) -> CourtComposition:
    """
    Find the court composition in the header of a decision
    :param header:          the string containing the header
    :param role_regexes:    the regexes for the court person roles
    :param namespace:       the namespace containing some metadata of the court decision
    :return:                the court composition
    """
    skip_strings = get_skip_strings()
    composition_strings = get_composition_strings(header)
    composition = CourtComposition()
    current_role = CourtRole.JUDGE
    last_person: CourtPerson = None
    person: CourtPerson = None
    last_gender = Gender.UNKNOWN

    # check if any of the role regexes can be found in the header. 
    # if there are none, we can exit this function.
    def any_matches():
        for gender in role_regexes:
            role_regex = role_regexes[gender]
            for regex_key in role_regex:
                regex = '|'.join(role_regex[regex_key])
                if re.search(regex, header):
                    return True
    if not any_matches():
        return

    for text in composition_strings:
        text = text.strip()
        # delete the last character if it's a dot following a lower-case character
        if re.search(r'[a-z]\.$', text):
            text=text[:-1]
        if len(text) == 0 or text in skip_strings[namespace['language']]:
            continue
        if (re.search(r'Vorsitz', text) or re.search(r'(?<![Vv]ize)[Pp]räsident', text)):  
        # Set president either to the current person or the last Person (case 1: Präsident Niklaus, case 2: Niklaus, Präsident)
            if last_person:
                composition.president = last_person
                continue
            else:
                pos = re.search(r'(?<![Vv]ize)[Pp]räsident(in)?', text)
                if pos == None:
                    pos = re.search(r'Vorsitz[\w]*', text)
                # assign gender depending on the noun ending
                if re.search(r'räsidentin', text):
                    last_gender = Gender.FEMALE
                elif re.search(r'räsident\b', text):
                    last_gender = Gender.MALE
                text = text[pos.span()[1]:]
                text = text.strip()
                composition.president = CourtPerson(text, last_gender)
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
                    # a name can consist of letters, periods, dashes, and spaces but starts with a letter
                    name_match = re.search(r'[A-Za-zÀ-ž]+[A-Za-zÀ-ž\.\- ]*', text[role_pos.span()[1] + 1:])

                    name = name_match.group() if name_match else text[role_pos.span()[1] + 1:]
                    if len(name.strip()) == 0:
                        if (last_role == CourtRole.CLERK and len(composition.clerks) == 0) or (last_role == CourtRole.JUDGE and len(composition.judges) == 0):
                            break

                        if len(composition.clerks) != 0:
                            last_person_name = composition.clerks.pop().name if (last_role == CourtRole.CLERK) else composition.clerks.pop().name # rematch in database with new role
                            last_person_new_match = CourtPerson(last_person_name, gender, court_role=current_role)
                            if current_role == CourtRole.JUDGE:
                                composition.judges.append(last_person_new_match)
                            elif current_role == CourtRole.CLERK:
                                composition.clerks.append(last_person_new_match)
                    matched_person = CourtPerson(name, gender, court_role=current_role)
                    if current_role == CourtRole.JUDGE and len(name.strip()) != 0:
                        composition.judges.append(matched_person)
                    elif current_role == CourtRole.CLERK and len(name.strip()) != 0:
                        composition.clerks.append(matched_person)
                    last_person = matched_person
                    last_gender = matched_person.gender
                    has_role_in_string = True
                    matched_gender_regex = True
                    break
        if not has_role_in_string:  # Current string has no role regex match
            # a name can consist of letters, periods, dashes, and spaces but starts with a letter
            name_match = re.search(r'[A-Za-zÀ-ž]+[A-Za-zÀ-ž\.\- ]*', text)
            if not name_match:
                continue
            name = name_match.group()
            person = CourtPerson(name, last_gender, court_role=current_role)
            matched_person = person
            if current_role == CourtRole.JUDGE:
                composition.judges.append(matched_person)
            elif current_role == CourtRole.CLERK:
                composition.clerks.append(matched_person)
            last_person = person
    return composition.toJSON() if composition else None

