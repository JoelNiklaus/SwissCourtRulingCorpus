from dataclasses import dataclass

from scrc.data_classes.person import Person
from scrc.enums.court_role import CourtRole
from scrc.enums.political_party import PoliticalParty


@dataclass
class CourtPerson(Person):
    court_role: CourtRole
    party: PoliticalParty
