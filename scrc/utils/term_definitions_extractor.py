from xml.etree import ElementTree as ET

import json

from root import ROOT_DIR
from scrc.utils.language_identification_singleton import LanguageIdentificationSingleton
from scrc.utils.log_utils import get_logger
from scrc.utils.xml_to_dict import XmlDictConfig, XmlListConfig


class TermDefinitionsExtractor:
    base_dir = ROOT_DIR / 'term_definitions'
    lang_id = LanguageIdentificationSingleton()
    languages = ['de', 'fr', 'it', 'rm', 'en']

    def __init__(self, ):
        self.logger = get_logger(__name__)

    def read_original_file(self, original_file):
        tree = ET.parse(original_file)
        root = tree.getroot()
        xmldict = XmlDictConfig(root)
        return xmldict['Eintraege']['Eintrag']

    def append_syns(self, syns, output, lang):
        if syns:  # make sure we don't append empty lists
            if lang in self.languages:
                output[lang].append(syns)

    def get_lang(self, syns):
        text = " ".join([syn['text'] for syn in syns])
        return self.lang_id.predict_lang(text)

    def extract_synonyms_stored_in_list_configs(self, language_dict):
        syns = []
        for lang_dict in language_dict:
            if isinstance(lang_dict, XmlListConfig):
                for synonym in lang_dict:
                    if 'Typ' in lang_dict:
                        syns.append({'type': synonym['Typ'], 'text': synonym['Text']})
        return syns

    def extract_synonyms_stored_in_dict_configs(self, language_dict):
        syns = []
        for lang_dict in language_dict:
            if isinstance(lang_dict, XmlDictConfig):
                if 'Typ' in lang_dict:
                    syns.append({'type': lang_dict['Typ'], 'text': lang_dict['Text']})
        return syns

    def extract_term_definitions(self):
        output_file = self.base_dir / 'term_definitions.json'
        if output_file.exists():
            raise ValueError(f"The file {output_file} exists already. Please delete it to rerun the extraction.")

        original_file = self.base_dir / 'ABR19_Titel_Bundeserlasse.xml'
        content = self.read_original_file(original_file)
        output = {lang: [] for lang in self.languages}
        for entry in content:
            languages_list = entry['Sprachzonen']['Sprachzone']
            for language_dict in languages_list:
                if isinstance(language_dict, XmlDictConfig):
                    synonyms = language_dict['Synonym']['Definition']

                    syns = self.extract_synonyms_stored_in_dict_configs(synonyms)
                    lang = language_dict['Sprache'].lower()
                    self.append_syns(syns, output, lang)

                    syns = self.extract_synonyms_stored_in_list_configs(synonyms)
                    lang = language_dict['Sprache'].lower()
                    self.append_syns(syns, output, lang)

                if isinstance(language_dict, XmlListConfig):
                    syns = self.extract_synonyms_stored_in_dict_configs(language_dict)
                    lang = self.get_lang(syns)
                    self.append_syns(syns, output, lang)

                    syns = self.extract_synonyms_stored_in_list_configs(language_dict)
                    lang = self.get_lang(syns)
                    self.append_syns(syns, output, lang)

        output_file.write_text(json.dumps(output))
        return output


if __name__ == '__main__':
    term_definitions_extractor = TermDefinitionsExtractor()
    output = term_definitions_extractor.extract_term_definitions()
