from enum import Enum


class Section(Enum):
    HEADER = 'header'
    FACTS = 'facts'
    CONSIDERATIONS = 'considerations'
    RULINGS = 'rulings'
    FOOTER = 'footer'

    @classmethod
    def without_facts(cls):
        return cls.HEADER, cls.CONSIDERATIONS, cls.RULINGS, cls.FOOTER