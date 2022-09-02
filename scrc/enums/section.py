from enum import Enum


class Section(Enum):
    FULLTEXT = 1
    HEADER = 2
    TOPIC = 3
    FACTS = 4
    CONSIDERATIONS = 5
    RULINGS = 6
    FOOTER = 7


    @classmethod
    def without_facts(cls):
        return cls.HEADER, cls.HEADER,cls.CONSIDERATIONS, cls.RULINGS, cls.FOOTER