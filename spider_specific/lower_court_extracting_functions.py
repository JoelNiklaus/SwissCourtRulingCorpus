from typing import Match, Optional, List, Tuple
from pathlib import Path
import re
import unicodedata
import json
import pandas as pd
from sqlalchemy.engine.base import Engine
from scrc.utils.main_utils import clean_text

"""
This file is used to extract the lower courts from decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
"""



def CH_BGer(header: str, namespace: dict, engine: Engine) -> Optional[str]:
    """
    Extract lower courts from decisions of the Federal Supreme Court of Switzerland
    :param header:     the string containing the header
    :param namespace:   the namespace containing some metadata of the court decision
    :return:            the sections dict
    """
    supported_languages = ['de', 'fr']
    if namespace['language'] not in supported_languages:
        message = f"This function is only implemented for the languages {supported_languages} so far."
        raise ValueError(message)


    file_number_regex = [r"""(?<=\() # find opening paranthesis without including it in result
        (?P<ID>[A-Z1-9]{0,4})
        (\.|\s)? # Split by point or whitespace or nothing
        (?P<YEAR>\d{2,4})
        (\.|\s)? # Split by point or whitespace or nothing
        (?P<NUMBER>\d{0,8})
        (?=\)) # find closing paranthesis without including it in result
    """, 
    r"""
    (?<=\() # find opening paranthesis without including it in result
    [A-Z]{1,2}
    (\-)?
    \d{2,4}
    (\/)?
    \d{0,4}
    (?=\)) # find closing paranthesis without including it in result
    """]

    information_start_regex = r'Beschwerde\sgegen\sd(?:as\sUrteil|en\s(Entscheid|Beschluss))\s|Instance précédente|recours'
    information_regex = {
        'court': [
            r'(\w*gericht(?=s[^\w]))',
            r'Tribunal .*?(?=[,\.]|du)',
            r'[Cc]our *?(?=[,\.]|du)'
        ],
        'canton': [
            r'((?<=des\s(?:Kantons\s))|((?<=des\s(?:Kantonsgerichts\s))))(Appenzell Innerrhoden|Appenzell Rhodes-Intérieures|Appenzello Interno)',
            r'((?<=des\s(?:Kantons\s))|((?<=des\s(?:Kantonsgerichts\s))))(Appenzell Ausserrhoden|Appenzell Rhodes-Extérieures|Appenzello Esterno)',
            r'((?<=des\s(?:Kantons\s))|((?<=des\s(?:Kantonsgerichts\s))))(Basilea Campagna|Basel-Land)',
            r'((?<=des\s(?:Kantons\s))|((?<=des\s(?:Kantonsgerichts\s))))(Basilea Città)',
            r'((?<=des\s(?:Kantons\s))|((?<=des\s(?:Kantonsgerichts\s))))(St\. Gallen|San Gallo)',
            r'((?<=des\s(?:Kantons\s))|((?<=des\s(?:Kantonsgerichts\s))\svon\s))[\wäöü-]*',
            r'((?<=des\s(?:Kantons\s))|((?<=des\s(?:Kantonsgerichts\s))))[\wäöü-]*',
            r'(?<=canton de )[\wéè]*',
            r'(?<=de l\'Etat de )[\wéè]*',
        ],
        'date': [
            r'(?P<DATE>(?P<DAY>\d?\d)\.?\s(?P<MONTH>\w{2,12})\s(?P<YEAR>\d{4}))'
        ],
        'chamber': [
            r'[IVX\d]+.\s\w*ammer',
            r'\w*ammer',
            r'[IVX\d]+.\s\w*our',
            r'[Cc]hambre.*?(?=[,\.]|du)',
            r'(?<![Rr]e)[Cc]our.*?(?=[,\.]|du)'
        ]
    }

     # combine multiple regex into one for each section due to performance reasons
    file_number_regex ='|'.join(file_number_regex)
    # normalize strings to avoid problems with umlauts
    file_number_regex = unicodedata.normalize('NFC', file_number_regex)
        
    def get_lower_court_file_number(header: str, namespace: dict) -> Optional[Match]:
        result = re.search(file_number_regex, header, re.X)
        return result

    def get_lower_court_by_file_number(file_number: Match) -> Optional[str]:
        languages = ['de', 'fr', 'it']
        chamber = None
        
        for lang in languages:
            if file_number.group('YEAR'):
                id = file_number.group('ID')
                year = file_number.group('YEAR')
                number = file_number.group('NUMBER')
                print(f"Special Case: \n{file_number.group()}\n")
                chamber = pd.read_sql(f"SELECT chamber FROM {lang} WHERE file_number LIKE '{id}%{year}%{number}'", engine.connect())
            else: 
                print(f"Normal Case: \n{file_number.group()}\n")
                chamber = pd.read_sql(f"SELECT chamber FROM {lang} WHERE file_number = '{file_number.group()}'", engine.connect())
            print(chamber)
        
        return None

    def prepareCantonForQuery(canton: str, court_chambers_data) -> str:
        for canton_short in court_chambers_data:
            current_canton = court_chambers_data[canton_short]
            if current_canton['de'] == canton or \
                current_canton['fr'] == canton or \
                current_canton['it'] == canton:
                return canton_short

    def prepareCourtForQuery(court: str, canton:str, court_chambers_data) -> str:
        canton_court_data = court_chambers_data[canton]
        for current_court_short in canton_court_data['gerichte']:
            current_court = canton_court_data['gerichte'][current_court_short]
            if current_court['de'] == court or \
                current_court['fr'] == court or \
                current_court['it'] == court:
                return current_court_short
        return court # court shortcut not found, then return original string

    def prepareDateForQuery(date: str) -> str:
        translation_dict = {
            "Januar": "Jan", "Februar": "Feb", "März": "Mar", "Mai": "May", "Juni": "June", "Juli": "July", "Oktober": "Oct", "Dezember": "Dec",
            "janvier": "Jan", "février": "Feb", "mars": "Mar", "juin": "june", "juillet": "July", "août": "Aug", "septembre": "Aept", "octobre": "Oct", "novembre": "Nov", "décembre": "Dec",
            "Gennaio": "Jan", "Febbraio": "Feb", "Marzo": "Mar", "Aprile": "Apr", "Maggio": "May", "Giugno": "June", "Luglio": "July", "Agosto": "Aug", "Settembre": "Sept", "Ottobre": "Oct", "Novembre": "Nov", "Dicembre": "Dec"
        }

        for k,v in translation_dict.items():
            date = date.replace(k, v)

        return pd.to_datetime(date, errors='ignore', dayfirst=True).strftime('%Y-%m-%d')
        

    def get_lower_court_by_date_and_court(lower_court_information) -> Optional[str]:
        languages = ['de', 'fr', 'it']
        chamber = None

        if 'canton' in lower_court_information:
            court_chambers_data = json.loads(Path("court_chambers.json").read_text())
            lower_court_information['canton'] = prepareCantonForQuery(lower_court_information['canton'], court_chambers_data)
            if 'court' in lower_court_information:
                lower_court_information['court'] = prepareCourtForQuery(lower_court_information['court'], lower_court_information['canton'], court_chambers_data)

        else:
            if 'court' in lower_court_information:
                court_chambers_data = json.loads(Path("court_chambers.json").read_text())
                lower_court_information['court'] = prepareCourtForQuery(lower_court_information['court'], 'CH', court_chambers_data)
                if re.match(r'CH_',lower_court_information['court']):
                    lower_court_information['canton'] = 'CH'
        lower_court_information['date'] = prepareDateForQuery(lower_court_information['date'])

        # try to find file in database
        """ for lang in languages:
            query = f"SELECT chamber, court, canton FROM {lang} WHERE date = '{prepareDateForQuery(lower_court_information['date'])}'"
            for item in lower_court_information:
                if item == 'date':
                    continue
                if lower_court_information[item] == None:
                    continue
                query += f" AND {item} = '{lower_court_information[item]}'"
            chamber = pd.read_sql(query, engine.connect())
            print(query, chamber, sep="\n") """
        
        return lower_court_information

    def get_court_information(header, namespace):
        result = {}
        start_pos = re.search(information_start_regex, header)
        if start_pos:
            header = header[start_pos.span()[0]:]
        
        for information_key in information_regex:
            regex = '|'.join(information_regex[information_key])
            # not a normal regex search so we find last occurence
            regex_result = None
            for regex_result in re.finditer(regex, header):
                pass
            if regex_result:
                result[information_key] = regex_result.group()
        
        return result

    # make sure we don't have any nasty unicode problems
    header = clean_text(header)
    header.replace('Appenzell I.Rh.', 'Appenzell Innerrhoden')
    header.replace('Appenzell A.Rh.', 'Appenzell Ausserrhoden')


    lower_court = None
    #lower_court_file_number = get_lower_court_file_number(header, namespace)
    lower_court_information = get_court_information(header, namespace)
    """ if lower_court_file_number:
        print(f'Got File number of previous court {lower_court_file_number} \n{header}\n{namespace["html_url"]}')
        lower_court = get_lower_court_by_file_number(lower_court_file_number)
        input() """
    if lower_court is None:
        try:
            lower_court = get_lower_court_by_date_and_court(lower_court_information)
            print(header, lower_court, sep="\n")
            input()
        except:
            return None
    return lower_court

# This needs special care
# def CH_BGE(rulings: str, namespace: dict) -> Optional[List[str]]:
#    return CH_BGer(rulings, namespace)
