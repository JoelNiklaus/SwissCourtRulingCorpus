from scrc.data_classes.law import Law
from scrc.enums.language import Language
from scrc.utils.log_utils import get_logger
from scrc.utils.term_definitions_converter import TermDefinitionsConverter


class LawUtilSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LawUtilSingleton, cls).__new__(cls)
            # Put any initialization here.
            cls._instance.init()
        return cls._instance

    def init(self):
        self.logger = get_logger(__name__)
        #  IMPORTANT: we need to take care of the fact that the laws are named differently in each language but refer to the same law!
        self.law_abbr_by_lang = self.build_law_abbr_by_lang()  # the abbreviations are the keys
        self.law_id_by_lang = {lang: {v: k for k, v in laws.items()}
                               for lang, laws in self.law_abbr_by_lang.items()}  # the ids are the keys

    @staticmethod
    def build_law_abbr_by_lang():
        term_definitions = TermDefinitionsConverter().extract_term_definitions()
        languages = [lang.value for lang in Language]
        law_abbr_by_lang = {lang: dict() for lang in languages}

        for definition in term_definitions:
            for lang in definition['languages']:
                if lang in languages:
                    for entry in definition['languages'][lang]:
                        if entry['type'] == 'ab':  # ab stands for abbreviation
                            # append the string of the abbreviation as key and the id as value
                            law_abbr_by_lang[lang][entry['text']] = definition['id']
        return law_abbr_by_lang

    def get_law_by_abbreviation(self, abbreviation: str) -> Law:
        """
        Retrieves a localized Law object from a law abbreviation
        :param abbreviation:
        :return:
        """
        for lang, laws in self.law_abbr_by_lang.items():
            if abbreviation in laws.keys():
                id = laws[abbreviation]
                abbreviations = {}
                for lang, abbrs_by_id in self.law_id_by_lang.items():
                    if id in abbrs_by_id:  # if we find the id
                        abbreviations[lang] = abbrs_by_id[id]  # add to the abbreviations dict
                return Law(id, abbreviations)
        raise ValueError(f"Please supply a valid law abbreviation. '{abbreviation}' could not be found")


if __name__ == '__main__':
    law_util = LawUtilSingleton()
    law = law_util.get_law_by_abbreviation("PSPV")
    print(law)
