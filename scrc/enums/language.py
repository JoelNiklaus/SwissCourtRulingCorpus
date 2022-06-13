from enum import Enum
from typing import Union


class Language(Enum):
    DE = 'de'
    FR = 'fr'
    IT = 'it'
    EN = 'en'  # maybe remove this
    RM = 'rm'
    UK = '--'
    
    @staticmethod
    def get_id_value(language_str) -> int:
        map_id = {
            'de': 1,
            'fr': 2,
            'it': 3,
            'en': 4
        }
        if language_str in map_id:
            return map_id.get(language_str)
        else: return -1
        