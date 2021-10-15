from dataclasses import dataclass
from typing import List

from scrc.data_classes.court_person import CourtPerson


@dataclass
class CourtComposition:
    president: CourtPerson
    judges: List[CourtPerson]
    clerks: List[CourtPerson]
