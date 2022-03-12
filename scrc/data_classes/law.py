from dataclasses import dataclass


@dataclass
class Law:
    sr_number: int
    abbreviations: dict = None  # localized abbreviation

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Law):
            return self.sr_number == other.sr_number
        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.sr_number)
