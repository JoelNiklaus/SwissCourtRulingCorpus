import unicodedata
import collections
from typing import Optional, List, Dict, Union

import bs4
import re
import json

from scrc.enums.language import Language
from scrc.enums.section import Section
from scrc.utils.main_utils import clean_text

"""
This file is used to extract sections from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
Overview of spiders still todo: https://docs.google.com/spreadsheets/d/1FZmeUEW8in4iDxiIgixY4g0_Bbg342w-twqtiIu8eZo/edit#gid=0
"""


def XX_SPIDER(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """
    # This is an example spider. Just copy this method and adjust the method name and the code to add your new spider.
    pass


def VD_Omni(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    # print("\nVD_Omni\n")

    def get_paragraphs(soup):
        """
        Get composition of the decision
        :param soup:
        :return:
        """
        # by some bruteforce I have found out that the html files in VD_Omni all have a single div with one of the
        # following classes:
        possible_main_div_classes = ["WordSection1", "Section1"]
        # Iterate over this list and find out the current decision is using which class type
        for main_div_class in possible_main_div_classes:

            div = soup.find_all("div", class_=main_div_class)
            if (len(div) != 0):
                class_found = main_div_class
                break
        # We expect to have only one main div
        assert len(div) == 1
        # If the divs is empty raise an error
        if (len(div) == 0):
            message = f"The main div has an unseen class type"
            raise ValueError(message)
        paragraphs = []
        # paragraph = None
        for element in div:
            if isinstance(element, bs4.element.Tag):
                text = str(element.string)
                # This is a hack to also get tags which contain other tags such as links to BGEs
                if text.strip() == 'None':
                    text = element.get_text()
                paragraph = clean_text(text)
                if paragraph not in ['', ' ', None]:  # discard empty paragraphs
                    paragraphs.append(paragraph)
        return paragraphs

    def get_composition_candidates(paragraphs, RegEx):
        composition_candidates = []
        # did we miss composition of some of the decisions?
        missed_cnt = 0
        for paragraph in paragraphs:
            composition = RegEx.search(paragraph)
            # If we did not find the RegEx in the paragraph
            if composition is None:
                missed_cnt += 1
                continue
            # If we found it, we will take a 2*range substring and use it as composition candidate
            starting_index = composition.start()
            # Hopefully the composition length will not be more than 400 characters
            range = 200
            # starting index
            s_indx = starting_index - range if (starting_index > range) else 0
            # ending index
            e_indx = starting_index + range if (len(paragraph) > starting_index + range) else len(paragraph)
            # extract the compostion
            composition_candidate = paragraph[s_indx:e_indx]
            # Store all the compositions in a signle list
            composition_candidates.append(composition_candidate)
        return composition_candidates, missed_cnt

    def find_the_keywords(candidate):
        """

                @param candidate: a 400 charachter string which should contain the judicial people
                @return: a dictionary with roles as keys and each value contains a tuple of 1)Boolean value indicating whether
                the role is availabe or not 2)the span of occurance of the role's keyword
                """
        pr_available = False
        cm_available = False
        as_available = False
        gr_available = False
        ju_available = False
        jusu_available = False
        ti_available = False
        cm_end_available = False

        pr_span = None
        cm_span = None
        as_span = None
        gr_span = None
        ju_span = None
        jusu_span = None
        ti_span_list = None
        cm_end_span = None

        pr_ = pr_RegEx.search(candidate)
        cm_ = cm_RegEx.search(candidate)
        as_ = as_RegEx.search(candidate)
        gr_ = gr_RegEx.search(candidate)
        ju_ = ju_RegEx.search(candidate)
        jusu_ = juSup_RegEx.search(candidate)
        ti_ = ti_RegEx.search(candidate)
        cm_end_ = cm_end_RegEx.search(candidate)

        if pr_ is not None:
            pr_available = True
            pr_span = pr_.span()

        if cm_ is not None:
            cm_available = True
            cm_span = cm_.span()

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
        if cm_end_ is not None:
            cm_end_available = True
            cm_end_span = cm_end_.span()

        keyword_dict = {'cm': [cm_available, cm_span], 'pr': [pr_available, pr_span],
                        'as': [as_available, as_span], 'gr': [gr_available, gr_span],
                        'ju': [ju_available, ju_span], 'jusu': [jusu_available, jusu_span],
                        'ti': [ti_available, ti_span_list], 'cm_end': [cm_end_available, cm_end_span]}
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

    def extraction_using_titles(ti_, cm_end_, candidate, keywords):
        """

        @param ti_: a list of title's span()
        @param cm_end_: output of cm_RegEx which is None if we were not able to find the end of the composition.
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
                    # if we have found the end of composition simply return what has remained
                    if (cm_end_ is not None):
                        roles.append([val[0], candidate[val[1][1][1]:]])
                    # if did not manage to find the end of the composition, limit the name to 25 characters
                    else:
                        roles.append([val[0], candidate[val[1][1][1]:val[1][1][1] + 20]])  # role_end to the role_end+25

                else:
                    message = f"We still do not support extraction if the name appears after the role keyword"
                    raise ValueError(message)

            # we need to store the prev_role for cases when several people have the same role
            prev_role = val[1][1][0]
            # increase the indexes for next loop
            idx_val += 1
            idx_title += 1
        return roles

    def extraction_using_composition(cm_, cm_end_, candidate, keywords):
        """

        @param cm_: The output of cm_RegEx. We can call its span() method to find where it occurs in the string
        @param cm_end_: The output of cm_end_RegEx. We can call its start() method to find where the composition ends
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

    def extract_judicial_people(candidates):
        """

        @param candidates: a list of all the candidates for compostion
        @return: a list of tuples with judicial people's 1) role 2)their details
        """
        roles = []
        # iterate over all the candidates
        for candidate in candidates:
            # find all the keyword's occurance in the candidate
            keywords = find_the_keywords(candidate)
            # pop out composition, end of composition, and titles
            cm_ = keywords.pop('cm')
            cm_end_ = keywords.pop('cm_end')
            ti_ = keywords.pop('ti')
            # it is vital to see if titles are used or not
            ti_available = ti_[0]
            ti_ = ti_[1]
            # preprocessing: cut the candidate when the composition end is known
            if cm_end_[0]:
                candidate = candidate[:cm_end_[1][0]]
            # preprocessing: eliminate those roles that were not found
            keywords = eliminate_invalid_roles(keywords)
            # preprocessing: sort the roles based on their order (which comes earlier?)
            sorted_keywords = sorted(keywords.items(), key=lambda e: e[1][1][0])
            # if decision is using the people titles
            if ti_available is not False and len(ti_) >= len(keywords.items()):

                roles.append(extraction_using_titles(ti_, cm_end_, candidate, sorted_keywords))

            # if titles are not used or not everybody has a title
            else:
                # we may be able to extract people using the composition keyword
                if cm_[0]:

                    roles.append(extraction_using_composition(cm_, cm_end_, candidate, sorted_keywords))

                else:
                    message = f"We still do not support extraction without tiles and 'composition' keyword"
                    raise ValueError(message)
        return roles

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
        roles_keys = {'pr': 'president', 'as': 'assesseur', 'gr': 'greffier', 'ju': 'juge', 'jusu': 'juge suppléante'}
        feminine_titles = ['Mme', 'Mme.', 'Mmes', 'Mlle', 'Mlle.']
        masculine_titles = ['M', 'M.', 'MM.', 'MM', 'Messieurs']
        plural_titles = ['Mmes', 'MM.', 'MM', 'Messieurs']
        separator_RegEx = re.compile(r'\bet\b')
        details = {}

        for idx_r, r in enumerate(roles[0]):

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
                        gender = 'female'
                    elif title in masculine_titles:
                        gender = 'male'
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

    paragraphs = get_paragraphs(decision)
    # Currently, we search the whole decision for the following keywords: president, presidence, compose, composition
    composition_RegEx = re.compile(r'[P,p]r[é,e]siden[t,c]|compos[é,e] | [C,c]omposition')
    composition_candidates, missed_cnt = get_composition_candidates(paragraphs, composition_RegEx)

    if missed_cnt != 0:
        # We write all the missed descisions to a file to take a closer look and improve the efficiency
        f = open("VD_Omni_missed_decisions.txt", "a")
        f.write('%s' % paragraphs)
        f.write('\n-------------------------------------------------------------\n')
        f.close()
        message = f"We have missed the judicial people for some decisions"
        raise ValueError(message)
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
    # find the end of composition
    cm_end_RegEx = re.compile(r'([E,e]n|les)? fait(s)?')

    roles = extract_judicial_people(composition_candidates)
    details = extract_details(roles)
    f = open("VD_Omni_judicial_person.txt", "a")
    f.write('%s' %details)
    f.write('\n-------------------------------------------------------------\n')
    f.close()



def CH_BGer(decision: Union[bs4.BeautifulSoup, str], namespace: dict) -> Optional[Dict[Section, List[str]]]:
    """
    :param decision:    the decision parsed by bs4 or the string extracted of the pdf
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict (keys: section, values: list of paragraphs)
    """

    print("\nCH_BGer\n")
    # As soon as one of the strings in the list (regexes) is encountered we switch to the corresponding section (key)
    # (?:C|c) is much faster for case insensitivity than [Cc] or (?i)c
    all_section_markers = {
        Language.DE: {
            # "header" has no markers!
            # at some later point we can still divide rubrum into more fine-grained sections like title, judges, parties, topic
            # "title": ['Urteil vom', 'Beschluss vom', 'Entscheid vom'],
            # "judges": ['Besetzung', 'Es wirken mit', 'Bundesrichter'],
            # "parties": ['Parteien', 'Verfahrensbeteiligte', 'In Sachen'],
            # "topic": ['Gegenstand', 'betreffend'],
            Section.FACTS: [r'Sachverhalt:', r'hat sich ergeben', r'Nach Einsicht', r'A\.-'],
            Section.CONSIDERATIONS: [r'Erwägung:', r'[Ii]n Erwägung', r'Erwägungen:'],
            Section.RULINGS: [r'erkennt d[\w]{2} Präsident', r'Demnach (erkennt|beschliesst)', r'beschliesst.*:\s*$',
                              r'verfügt(\s[\wäöü]*){0,3}:\s*$', r'erk[ae]nnt(\s[\wäöü]*){0,3}:\s*$',
                              r'Demnach verfügt[^e]'],
            Section.FOOTER: [
                r'^[\-\s\w\(]*,( den| vom)?\s\d?\d\.?\s?(?:Jan(?:uar)?|Feb(?:ruar)?|Mär(?:z)?|Apr(?:il)?|Mai|Jun(?:i)?|Jul(?:i)?|Aug(?:ust)?|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?|Dez(?:ember)?)\s\d{4}([\s]*$|.*(:|Im Namen))',
                r'Im Namen des']
        },
        Language.FR: {
            Section.FACTS: [r'Faits\s?:', r'en fait et en droit', r'(?:V|v)u\s?:', r'A.-'],
            Section.CONSIDERATIONS: [r'Considérant en (?:fait et en )?droit\s?:', r'(?:C|c)onsidérant(s?)\s?:',
                                     r'considère'],
            Section.RULINGS: [r'prononce\s?:', r'Par ces? motifs?\s?', r'ordonne\s?:'],
            Section.FOOTER: [
                r'\w*,\s(le\s?)?((\d?\d)|\d\s?(er|re|e)|premier|première|deuxième|troisième)\s?(?:janv|févr|mars|avr|mai|juin|juill|août|sept|oct|nov|déc).{0,10}\d?\d?\d\d\s?(.{0,5}[A-Z]{3}|(?!.{2})|[\.])',
                r'Au nom de la Cour'
            ]
        },
        Language.IT: {
            Section.FACTS: [r'(F|f)att(i|o)\s?:'],
            Section.CONSIDERATIONS: [r'(C|c)onsiderando', r'(D|d)iritto\s?:', r'Visto:', r'Considerato'],
            Section.RULINGS: [r'(P|p)er questi motivi'],
            Section.FOOTER: [
                r'\w*,\s(il\s?)?((\d?\d)|\d\s?(°))\s?(?:gen(?:naio)?|feb(?:braio)?|mar(?:zo)?|apr(?:ile)?|mag(?:gio)|giu(?:gno)?|lug(?:lio)?|ago(?:sto)?|set(?:tembre)?|ott(?:obre)?|nov(?:embre)?|dic(?:embre)?)\s?\d?\d?\d\d\s?([A-Za-z\/]{0,7}):?\s*$'
            ]
        }
    }

    if namespace['language'] not in all_section_markers:
        message = f"This function is only implemented for the languages {list(all_section_markers.keys())} so far."
        raise ValueError(message)

    section_markers = all_section_markers[namespace['language']]

    # combine multiple regex into one for each section due to performance reasons
    section_markers = dict(map(lambda kv: (kv[0], '|'.join(kv[1])), section_markers.items()))

    # normalize strings to avoid problems with umlauts
    for section, regexes in section_markers.items():
        section_markers[section] = unicodedata.normalize('NFC', regexes)
        # section_markers[key] = clean_text(regexes) # maybe this would solve some problems because of more cleaning

    def get_paragraphs(soup):
        """
        Get Paragraphs in the decision
        :param soup:
        :return:
        """
        divs = soup.find_all("div", class_="content")
        # we expect maximally two divs with class content
        assert len(divs) <= 2

        paragraphs = []
        heading, paragraph = None, None
        for element in divs[0]:
            if isinstance(element, bs4.element.Tag):
                text = str(element.string)
                # This is a hack to also get tags which contain other tags such as links to BGEs
                if text.strip() == 'None':
                    text = element.get_text()
                # get numerated titles such as 1. or A.
                if "." in text and len(text) < 5:
                    heading = text  # set heading for the next paragraph
                else:
                    if heading is not None:  # if we have a heading
                        paragraph = heading + " " + text  # add heading to text of the next paragraph
                    else:
                        paragraph = text
                    heading = None  # reset heading
                paragraph = clean_text(paragraph)
                if paragraph not in ['', ' ', None]:  # discard empty paragraphs
                    paragraphs.append(paragraph)
        return paragraphs

    paragraphs = get_paragraphs(decision)
    return associate_sections(paragraphs, section_markers, namespace)


def associate_sections(paragraphs: List[str], section_markers, namespace: dict):
    paragraphs_by_section = {section: [] for section in Section}
    current_section = Section.HEADER
    for paragraph in paragraphs:
        # update the current section if it changed
        current_section = update_section(current_section, paragraph, section_markers)

        # add paragraph to the list of paragraphs
        paragraphs_by_section[current_section].append(paragraph)
    if current_section != Section.FOOTER:
        message = f"({namespace['id']}): We got stuck at section {current_section}. Please check! " \
                  f"Here you have the url to the decision: {namespace['html_url']}"
        raise ValueError(message)
    return paragraphs_by_section


def update_section(current_section: Section, paragraph: str, section_markers) -> Section:
    if current_section == Section.FOOTER:
        return current_section  # we made it to the end, hooray!
    sections = list(Section)
    next_section_index = sections.index(current_section) + 1
    # consider all following sections
    next_sections = sections[next_section_index:]
    for next_section in next_sections:
        marker = section_markers[next_section]
        paragraph = unicodedata.normalize('NFC', paragraph)  # if we don't do this, we get weird matching behaviour
        if re.search(marker, paragraph):
            return next_section  # change to the next section
    return current_section  # stay at the old section

# This needs special care
# def CH_BGE(decision: Any, namespace: dict) -> Optional[dict]:
#    return CH_BGer(decision, namespace)
