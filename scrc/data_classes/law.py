from dataclasses import dataclass


@dataclass
class Law:
    id: int
    abbreviations: dict = None  # localized abbreviation

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Law):
            return self.id == other.id
        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))
