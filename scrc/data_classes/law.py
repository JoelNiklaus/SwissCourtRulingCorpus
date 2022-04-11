from dataclasses import dataclass


@dataclass
class Law:
    sr_number: str
    abbreviations: dict = None  # localized abbreviation

    def __repr__(self):
        self.__str__()

    def __str__(self):
        return f"{'/'.join(self.abbreviations.values())} (SR: {self.sr_number})"

    def __lt__(self, other):
        return self.sr_number < other.sr_number

    def __le__(self, other):
        return self.sr_number <= other.sr_number

    def __gt__(self, other):
        return self.sr_number > other.sr_number

    def __ge__(self, other):
        return self.sr_number >= other.sr_number

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Law):
            return self.sr_number == other.sr_number
        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.sr_number)
