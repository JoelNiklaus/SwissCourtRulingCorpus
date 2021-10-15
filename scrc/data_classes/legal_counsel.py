from dataclasses import dataclass

from scrc.data_classes.person import Person
from scrc.enums.legal_type import LegalType


@dataclass
class LegalCounsel(Person):
    legal_type: LegalType
    # we could add information regarding legal areas
