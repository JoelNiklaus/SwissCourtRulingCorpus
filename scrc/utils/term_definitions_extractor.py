import json
from collections import OrderedDict
from pprint import pprint

import xmltodict

from root import ROOT_DIR
from scrc.utils.log_utils import get_logger


class TermDefinitionsExtractor:
    base_dir = ROOT_DIR / 'term_definitions'
    languages = ['de', 'fr', 'it', 'rm', 'en', 'es']

    def __init__(self, ):
        self.logger = get_logger(__name__)

    @staticmethod
    def read_original_file(original_file, ):
        xmldict = xmltodict.parse(original_file.read_text())
        return xmldict['xml']['Eintraege']['Eintrag']

    def extract_term_definitions(self):
        output_file = self.base_dir / 'term_definitions.json'
        if output_file.exists():
            self.logger.info(f"The file {output_file} exists already. Please delete it to rerun the extraction.")
            return json.loads(output_file.read_text())

        original_file = self.base_dir / 'ABR19_Titel_Bundeserlasse.xml'
        content = self.read_original_file(original_file)

        terms = []
        for entry in content:
            metadata = entry['Kopf']
            if metadata['BearbeitungsStatus'] != 'Validiert':
                continue  # skip not validated status
            if int(metadata['ZuverlaessigkeitsCode']) < 3:  # increase to 4 or 5 for better confidence
                continue  # skip bad reliability ones
            term = OrderedDict({
                'id': int(entry['@Id']),
                'collection': metadata['Sammlung'],
                'areas': metadata['Sachgebiete'],
                'languages': {lang: [] for lang in self.languages}
            })
            languages_list = entry['Sprachzonen']['Sprachzone']
            for language_dict in languages_list:
                lang = language_dict['@Sprache'].lower()
                synonyms = language_dict['Synonym']
                # make sure that we get a list all the time
                synonyms = [synonyms] if isinstance(synonyms, OrderedDict) else synonyms
                for synonym in synonyms:
                    for definition in synonym['Definition']:
                        if isinstance(definition, OrderedDict):
                            term['languages'][lang].append({'type': definition['Typ'], 'text': definition['Text']})

            terms.append(term)

        output_file.write_text(json.dumps(terms))
        self.logger.info("Successfully extracted the term definitions.")
        return terms


if __name__ == '__main__':
    term_definitions_extractor = TermDefinitionsExtractor()
    output = term_definitions_extractor.extract_term_definitions()
    # pprint(output)
