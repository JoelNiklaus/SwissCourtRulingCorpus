from dataclasses import dataclass, field
from typing import List

from scrc.data_classes.court_person import CourtPerson


@dataclass
class CourtComposition:
    president: CourtPerson = None
    judges: List[CourtPerson] = field(default_factory=list)
    clerks: List[CourtPerson] = field(default_factory=list)
