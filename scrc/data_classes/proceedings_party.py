from dataclasses import dataclass
from typing import List

from scrc.data_classes.legal_counsel import LegalCounsel
from scrc.data_classes.person import Person
from scrc.enums.legal_type import LegalType


@dataclass
class ProceedingsParty(Person):
    legal_type: LegalType
    legal_counsel: List[LegalCounsel]
