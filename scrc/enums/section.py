from enum import Enum


class Section(Enum):
    FULLTEXT = 1
    HEADER = 2
    FACTS = 3
    CONSIDERATIONS = 4
    RULINGS = 5
    FOOTER = 6
    TOPIC = 7

    @classmethod
    def without_facts(cls):
        return cls.HEADER, cls.CONSIDERATIONS, cls.RULINGS, cls.FOOTER