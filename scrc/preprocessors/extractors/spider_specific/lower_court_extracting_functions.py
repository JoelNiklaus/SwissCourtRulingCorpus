from typing import Optional
from pathlib import Path
import re
import json
import pandas as pd
from scrc.utils.main_utils import clean_text

"""
This file is used to extract the lower courts from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
"""


def XX_SPIDER(header: str, namespace: dict) -> Optional[str]:
    # This is an example spider. Just copy this method and adjust the method name and the code to add your new spider.
    pass


# check if court got assigned shortcut: SELECT count(*) from de WHERE lower_court is not null and lower_court <> 'null' and lower_court::json#>>'{court}'~'[A-Z0-9_]{2,}';
def CH_BGer(header: str, namespace: dict) -> Optional[str]:
    """
    Extract lower courts from decisions of the Federal Supreme Court of Switzerland
    :param header:     the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict
    """

    information_start_regex = r'Vorinstanz|Beschwerden?\sgegen|gegen\sden\s(Entscheid|Beschluss)|gegen\sdas\sUrteil|Gegenstand|Instance précédente|recours|révision de|ricorso|ricorrente|rettifica'

    information_regex = {
        'court_string': [
            r'(\w*gericht(?=s?[^\w]))',
            r'(?P<high_prio>Tribunal .*?(?=[,\.]| du| de la République et canton))',
            r'(?<![Rr]e)[Cc]our .*?(?=[,\.]| du| de la République et canton)',
            r'Tribunale .*?(?=[,\.]| del Cantone)'
        ],
        'canton': [
            r'((?<=des\s(?:Kantons\s))|((?<=des\s(?:Kantonsgerichts\s))))(Appenzell Innerrhoden|Appenzell Rhodes-Intérieures|Appenzello Interno)',
            r'((?<=des\s(?:Kantons\s))|((?<=des\s(?:Kantonsgerichts\s))))(Appenzell Ausserrhoden|Appenzell Rhodes-Extérieures|Appenzello Esterno)',
            r'((?<=des\s(?:Kantons\s))|((?<=des\s(?:Kantonsgerichts\s))))Basel-Land',
            r'((?<=des\s(?:Kantons\s))|((?<=des\s(?:Kantonsgerichts\s))))(St(\.)?\s?Gallen|San Gallo)',
            r'((?<=des\s(?:Kantons\s))|((?<=des\s(?:Kantonsgerichts\svon\s))))[\wäöü-]*',
            r'((?<=des\s(?:Kantons\s))|((?<=des\s(?:Kantonsgerichts\s))))[\wäöü-]*',
            r'(?<=canton d[eu] )Bâle-(Ville|Campagne)',
            r'(?<=canton d[eu] )[\wéè]*',
            r'(?<=de l\'Etat de )[\wéè]*',
            r'((?<=del Cantone )|(?<=del Cantone di )|(?<=del Cantone dei ))(San Gallo)',
            r'((?<=del Cantone )|(?<=del Cantone di )|(?<=del Cantone dei ))(Appenzello (Interno|Esterno))',
            r'((?<=del Cantone )|(?<=del Cantone di )|(?<=del Cantone dei ))(Basilea (Città|Campagna))',
            r'(?<=del Cantone dei )[\wéè]*',
            r'(?<=del Cantone di )[\wéè]*',
            r'(?<=del Cantone del )[\wéè]*',
            r'(?<=del Cantone )[\wéè]*'

        ],
        'date': [
            r'(?P<DATE>(?P<DAY>\d?\d|1(re|er)|2e|3e|premier|première|deuxième|troisième|1°)\.?\s(?P<MONTH>\w{2,12})\s(?P<YEAR>\d{4}))'
        ],
        'chamber_string': [
            r'[IVX\d]+.\s\w*ammer',
            r'\w*ammer',
            r'[IVX\d]+.\s\w*our',
            r'(?P<high_prio>[Cc]hambre.*?(?=[,\.]| du| de la [Cc]our))',
            r'(?<![Rr]e)[Cc]our.*?(?=[,\.]| du| de la [Cc]our)',
            r'[Cc]orte.*?(?=[,\.]| del Tribunale| del Cantone)',
            r'[Cc]amera.*?(?=[,\.]| del Tribunale| del Cantone)',
            r'Abteilung\s[\dIVX]+',
            r'[IVX\d]+.\s(\w+\s)?Abteilung'
        ],
        'file_number': [
            r'(?P<ID>[A-Z0-9]{2,6})[\.\s\-]?(?P<YEAR>\d{2,4})[\.\s\-]?(?P<NUMBER>[\dA-Z\-]{2,8})(?=\))',
            # ex: AB12.2021.13
            r'[A-Z0-9]{1,4}([\.\-_\/\s])\d{1,8}[\.\/\-]?(\d{4}|[A-Z\/]+(\d+)?)',  # ex: AB-12/2021
            r'[A-Z0-9]{1,3}(\s|\.)?((([\d]{3,6})|\/)\s??){2,6}(-[A-Z])?'  # ex: 720 16 328 / 176
        ]
    }

    def prepareCantonForQuery(canton: str, court_chambers_data) -> str:
        for canton_short in court_chambers_data:
            current_canton = court_chambers_data[canton_short]
            if current_canton['de'] == canton or \
                    current_canton['fr'] == canton or \
                    current_canton['it'] == canton:
                return canton_short
        print(canton)

    def prepareCourtForQuery(court: str, canton: str, court_chambers_data) -> str:
        canton_court_data = court_chambers_data[canton]
        for current_court_short in canton_court_data['gerichte']:
            current_court = canton_court_data['gerichte'][current_court_short]
            if current_court['de'] == court or \
                    current_court['fr'] == court or \
                    current_court['it'] == court:
                return current_court_short

    def prepareChamberForQuery(chamber: str, court: str, canton: str, court_chambers_data) -> str:
        if court not in court_chambers_data[canton]['gerichte']:
            return chamber
        possible_labels = court_chambers_data[canton]['gerichte'][court]['kammern']
        for current_short in possible_labels:
            current_court_data = court_chambers_data[canton]['gerichte'][court]['kammern'][current_short]
            if {'de', 'fr', 'it'} <= current_court_data.keys():
                if chamber in current_court_data['de'] or \
                        chamber in current_court_data['fr'] or \
                        chamber in current_court_data['it']:
                    return current_short
                chamber_without_number = re.sub(r'[IV0-9]*.\s', '', chamber)
                if chamber_without_number in current_court_data['de'] or \
                        chamber_without_number in current_court_data['fr'] or \
                        chamber_without_number in current_court_data['it']:
                    return current_short

    def prepareDateForQuery(date: str) -> str:
        translation_dict = {
            "Januar": "Jan", "Februar": "Feb", "März": "Mar", "Mai": "May", "Juni": "June", "Juli": "July",
            "Oktober": "Oct", "Dezember": "Dec",
            "Janvier": "Jan", "Février": "Feb", "Mars": "Mar", "Avril": "April", "Juin": "june", "Juillet": "July",
            "Août": "Aug", "Septembre": "Sept", "Octobre": "Oct", "Novembre": "Nov", "Décembre": "Dec",
            "Gennaio": "Jan", "Febbraio": "Feb", "Marzo": "Mar", "Aprile": "Apr", "Maggio": "May", "Giugno": "June",
            "Luglio": "July", "Agosto": "Aug", "Settembre": "Sept", "Ottobre": "Oct", "Novembre": "Nov",
            "Dicembre": "Dec",
            "1er": "01", "1re": "01", "2e": "02", "3e": "03", "premier": "01", "première": "01", "deuxième": "02",
            "troisième": "03", "1°": "01"
        }

        for k, v in translation_dict.items():
            date = date.replace(k, v)
            date = date.replace(k.lower(), v)

        return pd.to_datetime(date, errors='ignore', dayfirst=True).strftime('%Y-%m-%d')

    def get_lower_court_by_date_and_court(lower_court_information) -> Optional[str]:
        languages = ['de', 'fr', 'it']
        chamber = None

        if 'canton' in lower_court_information:
            court_chambers_data = json.loads(Path("legal_info/court_chambers.json").read_text())
            lower_court_information['canton'] = prepareCantonForQuery(lower_court_information['canton'],
                                                                      court_chambers_data)
            if 'court_string' in lower_court_information and lower_court_information['canton'] is not None:
                lower_court_information['court'] = prepareCourtForQuery(lower_court_information['court_string'],
                                                                        lower_court_information['canton'],
                                                                        court_chambers_data)
        else:
            if 'court_string' in lower_court_information:
                court_chambers_data = json.loads(Path("legal_info/court_chambers.json").read_text())
                lower_court_information['court'] = prepareCourtForQuery(lower_court_information['court_string'], 'CH',
                                                                        court_chambers_data)
                if re.match(r'CH_', lower_court_information['court']):
                    lower_court_information['canton'] = 'CH'

        if {'canton', 'chamber_string', 'court'} <= lower_court_information.keys() and all(
                value is not None for value in
                [lower_court_information['chamber_string'], lower_court_information['court'],
                 lower_court_information['canton']]):
            lower_court_information['chamber'] = prepareChamberForQuery(lower_court_information['chamber_string'],
                                                                        lower_court_information['court'],
                                                                        lower_court_information['canton'], json.loads(
                    Path("legal_info/court_chambers.json").read_text()))
        if 'date' in lower_court_information:
            lower_court_information['date'] = prepareDateForQuery(lower_court_information['date'])

        return lower_court_information

    def get_court_information(header, namespace):
        result = {}
        start_pos = re.search(information_start_regex, header) or re.search(r', gegen|Beschwerdeführer', header)
        if start_pos:
            header = header[start_pos.span()[0]:]

        for information_key in information_regex:
            regex = '|'.join(information_regex[information_key])
            # not a normal regex search so we find last occurence
            regex_result = None
            for regex_result in re.finditer(regex, header):
                if 'high_prio' in regex_result.groupdict() and regex_result.group('high_prio') != None:
                    break
                pass
            if regex_result:
                if 'high_prio' in regex_result.groupdict() and regex_result.group('high_prio') != None:
                    result[information_key] = regex_result.group('high_prio')
                else:
                    result[information_key] = regex_result.group()

        return result

    # make sure we don't have any nasty unicode problems
    header = clean_text(header)
    header = header.replace('Appenzell I.Rh.', 'Appenzell Innerrhoden')
    header = header.replace('Appenzell A.Rh.', 'Appenzell Ausserrhoden')
    header = header.replace('Appenzell I. Rh.', 'Appenzell Innerrhoden')
    header = header.replace('Appenzell A. Rh.', 'Appenzell Ausserrhoden')
    header = header.replace('Waadt', 'Waadtland')
    header = header.replace('Basilea-Città', 'Basilea Città')

    lower_court = None
    # lower_court_file_number = get_lower_court_file_number(header, namespace)
    lower_court_information = get_court_information(header, namespace)
    """ if lower_court_file_number:
        print(f'Got File number of previous court {lower_court_file_number} \n{header}\n{namespace["html_url"]}')
        lower_court = get_lower_court_by_file_number(lower_court_file_number)
        input() """
    if lower_court is None:
        try:
            lower_court = get_lower_court_by_date_and_court(lower_court_information)
            # print(header, lower_court, sep="\n")
        except:
            return None
    return lower_court or None

# This needs special care
# def CH_BGE(rulings: str, namespace: dict) -> Optional[List[str]]:
#    return CH_BGer(rulings, namespace)
