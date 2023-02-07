from dataclasses import dataclass


@dataclass
class Law:
    sr_number: str
    abbreviations: []  # abbreviation are different for different languages
    uuids: []

    def __init__(self, sr_number, law_abbrs):
        self.sr_number = sr_number
        # get all uuid and abbreviations for this sr_number
        laws = law_abbrs[(law_abbrs.sr_number.str.strip() == sr_number)]  # cannot differ French and Italian
        if len(laws.index) == 0:
            # only include citations that we can find in our corpus
            raise ValueError(f"The abbreviation ({sr_number}) cannot be found.")
        self.abbreviations = []
        self.uuids = []
        for index, row in laws.iterrows():
            self.abbreviations.append(row.abbreviation)
            self.uuids.append(row.uuid)
        self.abbreviations = set(self.abbreviations)

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
