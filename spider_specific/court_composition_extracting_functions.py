from pathlib import Path
import re
import json
from typing import Optional, Tuple

"""
This file is used to extract the lower courts from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
"""

# check if court got assigned shortcut: SELECT count(*) from de WHERE lower_court is not null and lower_court <> 'null' and lower_court::json#>>'{court}'~'[A-Z0-9_]{2,}';
def CH_BGer(header: str, namespace: dict) -> Optional[str]:
    """
    Extract lower courts from decisions of the Federal Supreme Court of Switzerland
    :param header:     the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict
    """
    supported_languages = ['de', 'fr', 'it']
    if namespace['language'] not in supported_languages:
        message = f"This function is only implemented for the languages {supported_languages} so far."
        raise ValueError(message)

    information_start_regex = r'Besetzung|Bundesrichter|Composition( de la Cour:)?|Composizione|Giudic[ie] federal|composta'
    role_regexes = {
        'm' : {
            'judges': [r'Bundesrichter(?!in)', r'MM?\.(( et|,) Mmes?)? les? Juges?( fédéra(l|ux))?', r'[Gg]iudici federali'],
            'clerks': [r'Gerichtsschreiber(?!in)', r'Greffier[^\w\s]*', r'[Cc]ancelliere']
        },
        'f' : {
            'judges': [r'Bundesrichterin(nen)?', r'Mmes? l(a|es) Juges? (fédérales?)?', r'MMe et MM?\. les? Juges?( fédéra(l|ux))?', r'[Gg]iudice federal'],
            'clerks': [r'Gerichtsschreiberin(nen)?', r'Greffière.*Mme', r'[Cc]ancelliera']
        }
    }

    skip_strings = {
        'de': ['Einzelrichter', 'Konkurskammer', 'Beschwerdeführerin', 'Beschwerdeführer', 'Kläger', 'Berufungskläger'],
        'fr': ['Juge suppléant', 'en qualité de juge unique'],
        'it': ['Giudice supplente', 'supplente']
    }

    start_pos = re.search(information_start_regex, header)
    if start_pos:
        header = header[start_pos.span()[0]:]
    end_pos = {
        'de': re.search(r'.(?=(1.)?(Partei)|(Verfahrensbeteiligt))', header) or re.search('Urteil vom', header) or re.search(r'Gerichtsschreiber(in)?\s\w*.', header) or re.search(r'[Ii]n Sachen', header) or re.search(r'\w{2,}\.', header),
        'fr': re.search(r'.(?=(Parties|Participant))', header) or re.search(r'Greffi[eè]re? M(\w)*\.\s\w*.', header),
        'it': re.search(r'.(?=(Parti)|(Partecipant))', header) or re.search(r'[Cc]ancellier[ae]:?\s\w*.', header) or re.search(r'\w{2,}\.', header),
    }
    end_pos = end_pos[namespace['language']]
    if end_pos:
        header = header[:end_pos.span()[1]-1]

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

    personal_information_database = json.loads(Path("personal_information.json").read_text())

    def match_person_to_database(name: str, role: str, current_gender: str):
        results = []
        name = name.replace('.', '').strip()
        split_name = name.split()
        initial = False
        if len(split_name) > 1:
            initial = next((x for x in split_name if len(x) == 1), None)
            split_name = list(filter(lambda x: len(x) > 1, split_name))
        if role in personal_information_database:
            for subcategory in personal_information_database[role]:
                for cat_id in personal_information_database[role][subcategory]:
                    for person in personal_information_database[role][subcategory][cat_id]:
                        if set(split_name).issubset(set(person['name'].split())):
                            if not initial or re.search(rf'\s{initial.upper()}\w*', person['name']):
                                results.append(person)
        else:
            for existing_role in personal_information_database:
                person, match = match_person_to_database(name, existing_role, current_gender)
                if match:
                    results.append(person)
        if len(results) == 1:
            if 'gender' not in results[0]:
                results[0]['gender'] = current_gender
            return results[0], True
        return {'name': name, 'gender': current_gender}, False

    def prepareFrenchNameAndFindGender(name: str) -> Tuple[str, Optional[str]]:
        gender = None
        if name.find('M. ') > -1:
            name = name.replace('M. ', '')
            gender = 'm'
        elif name.find('Mme') > -1:
            name = name.replace('Mme ', '')
            gender = 'f'
        return name, gender

    besetzung = {}
    current_role = 'judges'
    last_person = ''
    last_gender = 'm'
    for text in besetzungs_strings:
        text = text.strip()
        if len(text) == 0 or text in skip_strings[namespace['language']]:
            continue
        if re.search(r'(?<![Vv]ice-)[Pp]r[äée]sid', text):
            if last_person:
                besetzung['president'] = last_person
                continue
            else:
                text = text.split()[-1]
                besetzung['president'] = text
        has_role_in_string = False
        matched_gender_regex = False
        for gender in role_regexes:
            if matched_gender_regex:
                break
            role_regex = role_regexes[gender]
            for regex_key in role_regex:
                regex = '|'.join(role_regex[regex_key])
                role_pos = re.search(regex, text)
                if role_pos:
                    last_role = current_role
                    current_role = regex_key
                    if current_role not in besetzung:
                        besetzung[current_role] = []
                    name_match = re.search(r'[A-Z][A-Za-z\-éèäöü\s]*(?= Urteil)|[A-Z][A-Za-z\-éèäöü\s]*(?= )', text[role_pos.span()[1]+1:])
                    name = name_match.group() if name_match else text[role_pos.span()[1]+1:]
                    if len(name.strip()) == 0:
                        if len(besetzung[last_role]) == 0:
                            break
                        last_person = besetzung[last_role].pop()['name'] #rematch in database with new role
                        last_person_new_match, _ = match_person_to_database(last_person, current_role, gender)
                        besetzung[current_role].append(last_person_new_match)
                    if namespace['language'] == 'fr':
                        name, found_gender = prepareFrenchNameAndFindGender(name)
                        gender = found_gender or gender
                    matched_person, _ = match_person_to_database(name, current_role, gender)
                    besetzung[current_role].append(matched_person)
                    last_person = matched_person['name']
                    last_gender = matched_person['gender']
                    has_role_in_string = True
                    matched_gender_regex = True
                    break
        if not has_role_in_string:
            if current_role not in besetzung:
                besetzung[current_role] = []
            if namespace['language'] == 'fr':
                    text, found_gender = prepareFrenchNameAndFindGender(text)
                    last_gender = found_gender or last_gender
            name_match = re.search(r'[A-Z][A-Za-z\-éèäöü\s]*(?= Urteil)|[A-Z][A-Za-z\-éèäöü\s]*(?= )|[A-Z][A-Za-z\-éèäöü\s]*', text)
            if not name_match:
                continue
            name = name_match.group()
            matched_person, _ = match_person_to_database(name, current_role, last_gender)
            besetzung[current_role].append(matched_person)
            last_person = name  
    return besetzung

# This needs special care
# def CH_BGE(rulings: str, namespace: dict) -> Optional[List[str]]:
#    return CH_BGer(rulings, namespace)
