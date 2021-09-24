from typing import Dict, List, Optional
import re
import json


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
        'm': [r'Beschwerdeführer(?!in)', r'Beschwerdegegner(?!in)', r'recourant(?!e)', r'intimés?(?!e)', r'ricorrente'],
        'f': [r'Beschwerdeführerin', r'Beschwerdegegnerin', r'recourantes?', r'intimées?']
    }

    lawyer_representation = {
        'm' : [r'Rechtsanwalt', r'Fürsprecher(?!in)', r'Advokat(?!in)', r'avocats?(?!e)', r'dall\'avv\.', r'l\'avv\.'],
        'f' : [r'Rechtsanwältin', r'Fürsprecherin', r'Advokatin', r'avocates?']
    }

    lawyer_name = {
        'de': r'((Dr\.\s)|(Prof\.\s))*[\w\séäöü\.]*?(?=(,)|(.$)|. Gegen| und)',
        'fr': r'(?<=Me\s)[\w\séèäöü\.\-]*?(?=,| et)|(?<=Mes\s)[\w\séèäöü\.\-]*?(?=,| et)|(?<=Maître\s)[\w\séèäöü\.\-]*?(?=,| et)',
        'it': r'(lic\.?\s?|iur\.?\s?|dott\.\s?)*[A-Z].*?(?=,)'
    }

    start_pos = re.search(information_start_regex, header) or re.search(r'Gerichtsschreiber.*?\.', header) or re.search(r'[Gg]reffi[eè]re?.*?\S{2,}?\.', header)
    if start_pos:
        header = header[start_pos.span()[1]:]
    end_pos = {
        'de': re.search(r'(?<=Beschwerdegegnerin).+?', header) or re.search(r'(?<=Beschwerdegegner).+?', header) or re.search(r'Gegenstand', header) or re.search(r'A\.\- ', header) or re.search(r'gegen das Urteil', header),
        'fr': re.search(r'Objet', header) or re.search(r'Vu', header),
        'it': re.search(r'Oggetto', header)
    }
    end_pos = end_pos[namespace['language']]
    if end_pos:
        header = header[:end_pos.span()[0]]
    
    representation_start = '|'.join(representation_start)
    for key in lawyer_representation:
        lawyer_representation[key] = '|'.join(lawyer_representation[key])
    for key in party_gender:
        party_gender[key] = '|'.join(party_gender[key])


    def search_lawyers(text: str) -> List[Dict]:
        lawyers = []
        for (gender, current_regex) in lawyer_representation.items():
            pos = re.search(current_regex, text)
            if pos:
                lawyer = {}
                if not namespace['language'] == 'it':
                    lawyer['gender'] = gender
                name_match = re.search(lawyer_name[namespace['language']], text[pos.span()[1]:])
                if name_match and not text[pos.span()[1]] == ',':
                    lawyer['name'] = name_match.group()
                else:
                    name_match = re.search(lawyer_name[namespace['language']], text[:pos.span()[0]])
                    lawyer['name'] = name_match.group() if name_match else None
                lawyer['type'] = 'natural person'
                lawyers.append(lawyer)
       
        return lawyers

    def add_representation(text: str, current_party: dict) -> Dict:
        representations = []
        start_positions = tuple(re.finditer(representation_start, text))
        if not start_positions:
            return current_party

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

            name_match = re.search(r'[A-Z][\w\s\.\-\']*(?=,)',current_text)
            if name_match:
                name = name_match.group()
                if name.startswith('Me'):
                    representations.append({'name': name[2:], 'type': 'natural person'})
                    continue
                representations.append({'name': name, 'type': 'legal entity'})
                continue
            name_match = re.search(r'[A-Z][\w\s\.\-\']*',current_text)
            if name_match:
                name = name_match.group()
                representations.append({'name': name, 'type': 'legal entity'})
                continue
        representations = [dict(t) for t in {tuple(d.items()) for d in representations}] #remove duplicates
        if len(representations) > 0:
            current_party['representation'] = representations
        return current_party

    def get_party(text: str) -> List:
        current_person = {}
        result = []
        try:
            current_person['name'] = re.search(r'[A-Z1-9].*?(?=(,)|(.$)| Beschwerde)', text).group().strip()
        except AttributeError:
            return result

        if re.match(r'[1-9IVX]+\.(?!_)', current_person['name']):
            people_string = re.split(r'[1-9IVX]+\. ', text)
            for string in people_string[1:]:
                result.extend(get_party(string))

            for idx in range(len(result)):
                if 'gender' in result[idx]:
                    del result[idx]['gender']
            return result
        if re.match(r'([A-Z]\.)?[A-Z]\._$', current_person['name']):
            for gender, current_regex in party_gender.items():
                if re.search(current_regex, text):
                    if not namespace['language'] == 'it':
                        current_person['gender'] = gender
                    current_person['type'] = 'natural person'
                    result.append(current_person)
                    return result
            current_person['type'] = 'natural person'
            result.append(current_person)
            return result
        current_person['type'] = 'legal entity'
        result.append(current_person)
        return result

    party = {0: {
        'party': []
    }, 1: {
        'party': []
    }}
    header_parts = re.split('|'.join(second_party_start_regex), header)
    if len(header_parts) < 2:
        raise ValueError(f"({namespace['id']}): Header malformed for: {namespace['html_url']}")
    party[0] = add_representation(header_parts[0], party[0])
    party[1] = add_representation(header_parts[1], party[1])
    

    party[0]['party'] = get_party(header_parts[0])
    party[1]['party'] = get_party(header_parts[1])
    return json.dumps(party)

# This needs special care
# def CH_BGE(rulings: str, namespace: dict) -> Optional[List[str]]:
#    return CH_BGer(rulings, namespace)
