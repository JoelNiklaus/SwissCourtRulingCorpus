""" 
    This file can be used to recreat the setup_values.sql file.
    To recreate the whole file run the main function, to recreate only one table run and PRINT the function with the table name as the name.
"""
import json
from pathlib import Path

INSERT_STMT_PLACEHOLDER = 'INSERT INTO %s VALUES \n\t%s;'

LANGUAGES = ['de', 'fr', 'it', 'en']
JSON_FILE_LOADED = {} # Automatically gets filled via court_chambers_extended.json
CANTONS = [] # Automatically gets filled via court_chambers_extended.json
JUDGMENTS = ['approval', 'dismissal', 'inadmissible', 'partial_approval', 'partial_dismissal', 'unification', 'write_off']
CITATION_TYPES = ['ruling', 'law', 'commentary']
SECTION_TYPES = ['full_text', 'header', 'facts', 'considerations', 'rulings', 'footer']
JUDICIAL_PERSON_TYPES = ['federal_judge', 'deputy_federal_judge', 'clerk']
PARTY_TYPE = ['plaintiff', 'defendant', 'representation_plaintiff', 'representation_defendant']

def read_court_chambers_extended():
    with open(Path('../court_chambers_extended.json'), 'r') as file:
        data = json.load(file)
        global CANTONS
        CANTONS = list(data.keys())
    return data

JSON_FILE_LOADED = read_court_chambers_extended()


def language():
    languages = [f"(\'{language}\')" for language in LANGUAGES]
    return INSERT_STMT_PLACEHOLDER % ("\"language\"(iso_code)", ',\n\t'.join(languages))

def canton():
    cantons = [f"(\'{canton}\')" for canton in CANTONS]
    return INSERT_STMT_PLACEHOLDER % ("canton(short_code)", ',\n\t'.join(cantons))
    
def canton_name():
    # ((SELECT canton_id FROM canton WHERE short_code = 'CH'), (SELECT language_id FROM language WHERE iso_code ='de'),'Eidgenossenschaft'),
    VALUE_PLACEHOLDER = "((SELECT canton_id FROM canton WHERE short_code = '%s'), (SELECT language_id FROM language WHERE iso_code ='%s'),'%s')"
    values = []
    for language in LANGUAGES:
        for canton in CANTONS:
            if language in JSON_FILE_LOADED[canton]:
                canton_name = JSON_FILE_LOADED[canton][language]
            elif 'de' in JSON_FILE_LOADED[canton]:
                canton_name = JSON_FILE_LOADED[canton]['de']
            else:
                raise IndexError()
            values.append(VALUE_PLACEHOLDER % (canton, language, canton_name))
    return INSERT_STMT_PLACEHOLDER % ("canton_name(canton_id, language_id, \"name\")", ',\n\t'.join(values))
    
def spider():
    spiders = set()
    for canton in CANTONS:
        for court in JSON_FILE_LOADED[canton]['gerichte']:
            court_dict = JSON_FILE_LOADED[canton]['gerichte'][court]
            for chamber in court_dict['kammern']:
                chamber_dict = court_dict['kammern'][chamber]
                spiders.add(chamber_dict['spider'])
                    
    values = [f"('{spider}')" for spider in spiders]
    return INSERT_STMT_PLACEHOLDER % ('spider("name")', ',\n\t'.join(sorted(values)))

def court():
    # ((SELECT canton_id FROM canton WHERE short_code = 'AG'), 'AG_AK'),
    courts = []
    for canton in CANTONS:
        for court in JSON_FILE_LOADED[canton]['gerichte']:
            courts.append(f"((SELECT canton_id FROM canton WHERE short_code = '{canton}'), '{court}')")
    return INSERT_STMT_PLACEHOLDER % ('court(canton_id, court_string)', ',\n\t'.join(sorted(courts)))
   
def court_name():
    # ((SELECT court_id FROM court WHERE court_string = 'AG_AK'), (SELECT language_id FROM language WHERE iso_code = 'de'), 'Anwaltskommission')
    VALUE_PLACEHOLDER = "((SELECT court_id FROM court WHERE court_string = '%s'), (SELECT language_id FROM language WHERE iso_code = '%s'), '%s')"
    values = []
    for canton in CANTONS:
        for court in JSON_FILE_LOADED[canton]['gerichte']:
            court_dict = JSON_FILE_LOADED[canton]['gerichte'][court]
            for language in LANGUAGES:
                if language in court_dict:
                    court_name = court_dict[language]
                elif 'de' in court_dict:
                    court_name = court_dict['de']
                else:
                    raise IndexError()
                values.append(VALUE_PLACEHOLDER % (court, language, court_name.replace("'", "''")))
    
    return INSERT_STMT_PLACEHOLDER % ('court_name(court_id, "language_id", "name")', ',\n\t'.join(sorted(values)))

def chamber():
    # ((SELECT court_id FROM court WHERE court_string = 'AG_AK'), (SELECT spider_id FROM spider WHERE name = 'AG_Gerichte'), 'AG_AK_001')
    VALUE_PLACEHOLDER = "((SELECT court_id FROM court WHERE court_string = '%s'), (SELECT spider_id FROM spider WHERE name = '%s'), '%s')"
    values = []
    for canton in CANTONS:
        for court in JSON_FILE_LOADED[canton]['gerichte']:
            court_dict = JSON_FILE_LOADED[canton]['gerichte'][court]
            for chamber in court_dict['kammern']:
                spider = court_dict['kammern'][chamber]['spider']
                values.append(VALUE_PLACEHOLDER % (court, spider, chamber))
    return INSERT_STMT_PLACEHOLDER % ('chamber(court_id, spider_id, chamber_string)', ',\n\t'.join(sorted(values)))

def judgment():
    judgments = [f"(\'{judgment}\')" for judgment in JUDGMENTS]
    return INSERT_STMT_PLACEHOLDER % ("judgement(\"text\")", ',\n\t'.join(judgments))

def citation_type():
    citation_types = [f"(\'{citation_type}\')" for citation_type in CITATION_TYPES]
    return INSERT_STMT_PLACEHOLDER % ("citation_type(citation_type_name)", ',\n\t'.join(citation_types))

def section_type():
    section_types = [f"(\'{section_type}\')" for section_type in SECTION_TYPES]
    return INSERT_STMT_PLACEHOLDER % ("section_type(\"name\")", ',\n\t'.join(section_types))

def judicial_person_type():
    judicial_person_types = [f"(\'{judicial_person_type}\')" for judicial_person_type in JUDICIAL_PERSON_TYPES]
    return INSERT_STMT_PLACEHOLDER % ("judicial_person_type(\"name\")", ',\n\t'.join(judicial_person_types))

def party_type():
    party_types = [f"(\'{party_type}\')" for party_type in PARTY_TYPE]
    return INSERT_STMT_PLACEHOLDER % ("party_type(\"name\")", ',\n\t'.join(party_types))
  
def main():
    print(language())
    print(canton())
    print(canton_name())
    print(spider())
    print(court())
    print(court_name())
    print(chamber())
    print(judgment())
    print(citation_type())
    print(section_type())
    print(judicial_person_type())
    print(party_type())