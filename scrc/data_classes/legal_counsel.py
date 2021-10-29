from dataclasses import dataclass

from scrc.data_classes.person import Person
from scrc.enums.legal_type import LegalType


@dataclass
class LegalCounsel(Person):
    legal_type: LegalType = None
    # we could add information regarding legal areas

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, LegalCounsel):
            return False
        return self.legal_type == other.legal_type and self.name == other.name and self.gender == other.gender

    def __neq__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        if self.legal_type:
            return hash((self.legal_type.value, self.name, self.gender))
        return hash((self.name, self.gender))
        
            
