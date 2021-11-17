from pathlib import Path
import re
import json
from typing import Optional, Tuple
from scrc.data_classes.court_composition import CourtComposition
from scrc.data_classes.court_person import CourtPerson

from scrc.enums.court_role import CourtRole
from scrc.enums.gender import Gender
from scrc.enums.language import Language

"""
This file is used to extract the judicial persons from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
Overview of spiders still todo: https://docs.google.com/spreadsheets/d/1FZmeUEW8in4iDxiIgixY4g0_Bbg342w-twqtiIu8eZo/edit#gid=0
"""


def XX_SPIDER(header: str, namespace: dict) -> Optional[str]:
    # This is an example spider. Just copy this method and adjust the method name and the code to add your new spider.
    pass
def VD_Omni(header: str, namespace: dict) -> Optional[str]:


    def find_the_keywords(candidate):
        """

                @param candidate: a 400 charachter string which should contain the judicial people
                @return: a dictionary with roles as keys and each value contains a tuple of 1)Boolean value indicating whether
                the role is availabe or not 2)the span of occurance of the role's keyword
                """
        pr_available = False
        as_available = False
        gr_available = False
        ju_available = False
        jusu_available = False
        ti_available = False


        pr_span = None
        as_span = None
        gr_span = None
        ju_span = None
        jusu_span = None
        ti_span_list = None


        pr_ = pr_RegEx.search(candidate)
        as_ = as_RegEx.search(candidate)
        gr_ = gr_RegEx.search(candidate)
        ju_ = ju_RegEx.search(candidate)
        jusu_ = juSup_RegEx.search(candidate)
        ti_ = ti_RegEx.search(candidate)


        if pr_ is not None:
            pr_available = True
            pr_span = pr_.span()

        if as_ is not None:
            as_available = True
            as_span = as_.span()

        if gr_ is not None:
            gr_available = True
            gr_span = gr_.span()

        if ju_ is not None:
            ju_available = True
            ju_span = ju_.span()
        if jusu_ is not None:
            jusu_available = True
            jusu_span = jusu_.span()
        if ti_ is not None:
            ti_available = True
            ti_span_list = [m.span() for m in re.finditer(ti_RegEx, candidate)]

        keyword_dict = {
                        'pr': [pr_available, pr_span],
                        'as': [as_available, as_span],
                        'gr': [gr_available, gr_span],
                        'ju': [ju_available, ju_span],
                        'jusu': [jusu_available, jusu_span],
                        'ti': [ti_available, ti_span_list]
                        }
        return keyword_dict

    def eliminate_invalid_roles(keyword_dict):
        """

        @param keyword_dict: receives a dictionary with keys = role and values = tuples of bool (availability of the role)
        and the span of the keyword
        @return: returns a dictionary of roles whose availability field is True
        """
        invalid_roles = []
        for k, v in keyword_dict.items():
            # if a role is not found, mark it as invalid
            if not v[0]:
                invalid_roles.append(k)
        for k in invalid_roles:
            keyword_dict.pop(k)
        return keyword_dict

    def extraction_using_titles(ti_, candidate, keywords):
        """

        @param ti_: a list of title's span()
        @param candidate: a 400-character string which is candidate for containing the composition
        @param keywords: a list of sorted roles and their span()
        @return:
        """
        roles = []

        idx_val = 0
        idx_title = 0
        prev_role = None
        for val in keywords:

            ti_e = ti_[idx_title]
            if prev_role is not None:
                # Several people have the same role
                while ti_e[0] < prev_role and idx_title < len(ti_) - 1:
                    # I should go to next title
                    idx_title += 1
                    ti_e = ti_[idx_title]
            # the person's name and title are mentioned before their roles
            if (ti_e[0] < val[1][1][0]):
                roles.append([val[0], candidate[ti_e[0]:val[1][1][0]]])  # ti_start to role start
            # the person's name and title are mentioned after their roles
            else:
                # Is it the last item?
                if idx_val == len(keywords) - 1:
                    roles.append([val[0], candidate[val[1][1][1]:]])
                else:
                    message = f"We still do not support extraction if the name appears after the role keyword"
                    raise ValueError(message)

            # we need to store the prev_role for cases when several people have the same role
            prev_role = val[1][1][0]
            # increase the indexes for next loop
            idx_val += 1
            idx_title += 1
        return roles

    def extraction_using_composition(cm_, candidate, keywords):
        """
        @param cm_: The output of cm_RegEx. We can call its span() method to find where it occurs in the string
        @param candidate: a 400-character string which potentially contains the composition
        @param keywords: a sorted list of roles and their span
        @return: A list of tuples whith 1)the role keyword 2) their details
        """
        roles = []
        idx_val = 0
        prev_role = None
        next_role = None
        # for role in keywords
        for val in keywords:
            # If we are not at the end, store the next role in next_role variable
            if idx_val != len(keywords) - 1:
                next_role = keywords[idx_val + 1]
            # if composition_end is within president_start+5 then names come after their roles
            if (cm_[1][1] + 5 > val[1][1][0]):
                # if we are not at the end
                if (idx_val != len(keywords) - 1):
                    roles.append(
                        [val[0], candidate[val[1][1][1]:next_role[1][1][0]]])  # current_role_end to next_role_start
                # then we assume the last person details is just 25 characters
                else:
                    roles.append([val[0], candidate[val[1][1][1]:role[1][1][1] + 25]])
            # otherwise, the names appear before the keyword
            else:
                # we have just started
                if idx_val == 0:
                    roles.append([val[0], candidate[cm_[1][1]:val[1][1][0]]])  # composition_end to role_begin
                # if we are in between
                elif idx_val != len(keywords) - 1:
                    roles.append(
                        [val[0], candidate[prev_role[1][1][1]:val[1][1][0]]])  # prev_role_end to current_role_start
                else:
                    # some times we have 'Greffiere: name'
                    if candidate[val[1][1][1] + 1] == ':' or candidate[val[1][1][1] + 2] == ':':
                        roles.append([val[0], candidate[val[1][1][1]:val[1][1][1] + 25]])
                    else:
                        roles.append(
                            [val[0], candidate[prev_role[1][1][1]:val[1][1][0]]])  # prev_role_end to current_role_start

            # we need to store the prev_role for cases when one role has several titles
            prev_role = val
            # increase the indexes for next loop
            idx_val += 1
        return roles

    def extract_judicial_people(candidate):
        """

        @param candidates: the string candidates for compostion
        @return: a list of tuples with judicial people's 1) role 2)their details
        """

        # iterate over all the candidates

        # find all the keyword's occurance in the candidate
        keywords = find_the_keywords(candidate)
        # pop out composition, and titles
        cm_ = keywords.pop('cm')
        ti_ = keywords.pop('ti')
        # it is vital to see if titles are used or not
        ti_available = ti_[0]
        ti_ = ti_[1]
        # preprocessing: eliminate those roles that were not found
        keywords = eliminate_invalid_roles(keywords)
        # preprocessing: sort the roles based on their order (which comes earlier?)
        sorted_keywords = sorted(keywords.items(), key=lambda e: e[1][1][0])
        # if decision is using the people titles
        if ti_available is not False and len(ti_) >= len(keywords.items()):

            role = extraction_using_titles(ti_, candidate, sorted_keywords)

        # if titles are not used or not everybody has a title
        else:
            # we may be able to extract people using the composition keyword
            if cm_[0]:
                role = extraction_using_composition(cm_, candidate, sorted_keywords)

            else:
                message = f"We still do not support extraction without tiles and 'composition' keyword"
                raise ValueError(message)
        return role

    def remove_special_characters(in_str):
        """

        @param in_str: Input string
        @return: output string without the following characters: {",",";","s:","e:","******"}
        """
        # for ch in characters_to_replace:
        in_str = in_str.replace(",", "")
        in_str = in_str.replace(";", "")
        # some times roles are female or plural
        in_str = in_str.replace("s:", "")
        in_str = in_str.replace("e:", "")
        # remove sequences of '*'
        star_RegEx = re.compile(r'\*+')
        star_ = star_RegEx.search(in_str)
        if star_ is not None:
            star_s = star_.span()
            in_str = in_str[:star_s[0] - 1] + in_str[star_s[1] + 1:]
        return in_str

    def first_last_name(full_name, detail):
        """

        @param full_name: A string that potentially contains: first name (or initials) and last name
        @param detail: a dictionary which contains the details of the judicial people
        @return: the detail dictionary with two new records: first name (or initials) and last name
        """
        dot_RegEx = re.compile(r'\w\.\b')
        dot_ = dot_RegEx.search(full_name)
        names = full_name.split(" ")
        last_name = False
        # remove all the empty strings or strings of length 1
        for idx, name in enumerate(names):
            if name == "" or len(name) == 1:
                names.pop(idx)
        for name in names:
            # is it intials?
            if (dot_ is not None):
                detail['initials']: name
                last_name = True
            # is it the first or the last name?
            # In some decisions, we only have the last name
            elif (len(names) > 1 and last_name == False):
                detail['first name'] = name
                last_name = True
            else:
                detail['last name'] = name
        return

    def extract_details(roles):
        """

        @param roles: a list of tuples with judicial people's 1) role 2) details
        @return: A dictionary ready to be written in the json file
        """
        roles_keys = {'pr': CourtRole.PRESIDENT, 'as': CourtRole.ASSESSEUR, 'gr': CourtRole.CLERK,
                      'ju': CourtRole.JUDGE, 'jusu': CourtRole.JUDGE}
        feminine_titles = ['Mme', 'Mme.', 'Mmes', 'Mlle', 'Mlle.']
        masculine_titles = ['M', 'M.', 'MM.', 'MM', 'Messieurs']
        plural_titles = ['Mmes', 'MM.', 'MM', 'Messieurs']
        separator_RegEx = re.compile(r'\bet\b')
        details = {}

        for idx_r, r in enumerate(roles):

            detail_list = []
            key = roles_keys.get(r[0])
            # if they have used titles, our job is very easy
            ti_list = [m.span() for m in re.finditer(ti_RegEx, r[1])]
            #Are there several people with the same role?
            if len(ti_list) > 0:

                for ti_ in ti_list:
                    # body of the extracted record
                    body = r[1][
                           ti_[1] + 1:]  # I am using ti_[1]+1 to eliminate '.'s at the begging of names (when they have
                    # used MM instead of MM., for example.

                    # male or female?
                    gender = None
                    # what is the title?
                    title = r[1][ti_[0]:ti_[1]]

                    # set the gender
                    if title in feminine_titles:
                        gender = Gender.FEMALE
                    elif title in masculine_titles:
                        gender = Gender.MALE
                    else:
                        message = f"Undefined title: " + title
                        raise ValueError(message)

                    # Is it a plural title?
                    if title in plural_titles:

                        # I need to split the names, if they have used et
                        if separator_RegEx.search(body) is not None:
                            all_names = body.split('et')
                        else:
                            message = f"seperator is not et: " + body
                            raise ValueError(message)
                        for name in all_names:
                            detail = {}
                            name = remove_special_characters(name)
                            detail['gender'] = gender
                            first_last_name(name, detail)
                            detail['full name'] = name
                            detail_list.append(detail)

                    else:
                        detail = {}
                        # do we have multiple people?
                        if separator_RegEx.search(body) is not None:
                            body = body[:separator_RegEx.search(body).span()[0]]  # body [: until next title's start]

                        body = remove_special_characters(body)

                        detail['gender'] = gender
                        first_last_name(body, detail)
                        detail['full name'] = body
                        detail_list.append(detail)

            else:
                # what we will write to the json
                body = r[1]
                all_names = [body]
                if separator_RegEx.search(body) is not None:
                    all_names = body.split('et')
                for name in all_names:
                    detail = {}
                    name = remove_special_characters(name)
                    detail['gender'] = 'unknown'
                    first_last_name(name, detail)
                    detail['full name'] = name
                    detail_list.append(detail)
            details[key] = detail_list
        return details


    # Here are the RegEx for finding different roles and titles in the composition
    # composition
    cm_RegEx = re.compile(r'[C,c]omposition')
    # president
    pr_RegEx = re.compile(r'[P,p]r[é,e]siden[t,c]')
    # assesseur
    as_RegEx = re.compile(r'[A,a]ssesseur')
    # greffier
    gr_RegEx = re.compile(r'[G,g]reffi[e,è]r')
    # juges
    ju_RegEx = re.compile(r'[J,j]ug')
    # juge suppléante
    juSup_RegEx = re.compile(r'[J,j]uge\bsuppl[é,e]ant')
    # title
    ti_RegEx = re.compile(r'\bMme(\.)?\b|\bM(\.)?\b|\bMM(\.)?\b|Mlle(\.)?|Mme(s)?|Messieur(s)?')


    role = extract_judicial_people(composition_candidates)
    details = extract_details(role)
    return details


# check if court got assigned shortcut: SELECT count(*) from de WHERE lower_court is not null and lower_court <> 'null' and lower_court::json#>>'{court}'~'[A-Z0-9_]{2,}';
def CH_BGer(header: str, namespace: dict) -> Optional[str]:
    """
    Extract judicial persons from decisions of the Federal Supreme Court of Switzerland
    :param header:      the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict
    """

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

    skip_strings = {
        Language.DE: ['Einzelrichter', 'Konkurskammer', 'Beschwerdeführerin', 'Beschwerdeführer', 'Kläger', 'Berufungskläger'],
        Language.FR: ['Juge suppléant', 'en qualité de juge unique'],
        Language.IT: ['Giudice supplente', 'supplente']
    }

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

    personal_information_database = json.loads(Path("personal_information.json").read_text())

    def match_person_to_database(person: CourtPerson, current_gender: Gender) -> Tuple[CourtPerson, bool]:
        """"Matches a name of a given role to a person from personal_information.json"""
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
                                    person.gender = Gender(db_person['party'])
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

