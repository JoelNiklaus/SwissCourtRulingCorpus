from scrc.data_classes.citation import Citation
from operator import attrgetter


class RulingCitation(Citation):
    """ For more information check wikipedia: https://de.wikipedia.org/wiki/Entscheidungen_des_Schweizerischen_Bundesgerichts"""
    year: int  # 1 == 1875, 121 == 1995, 129 == 2003, 147 == 2021, 150 == 2024
    volume: str  # roman numeral, not using the roman numeral class because of old volumes such as Ia
    page_number: int

    # compare first by year, then by volumne and finally by page_number
    comparison_attributes = attrgetter("year", "volume", "page_number")

    def __init__(self, citation_str, language):
        self.language = language
        if language == "de":
            self.ruling_str = "BGE"
        elif language == "fr":
            self.ruling_str = "ATF"
        elif language == "it":
            self.ruling_str = "DTF"
        if citation_str[0].isnumeric():  # 'BGE' is missing at the beginning
            citation_str = self.ruling_str + " " + citation_str  # prepend 'BGE '
        parts = citation_str.split(" ")
        try:
            self.year = int(parts[1])
            self.volume = parts[2]
            self.page_number = int(parts[3])
        except ValueError:
            raise ValueError(f"The Citation String ({citation_str}) could not be parsed successfully.")

    def __str__(self):
        return f"{self.ruling_str} {self.year} {self.volume} {self.page_number}"

    def cit_string(self):
        return f"{self.ruling_str}-{self.year}-{self.volume}-{self.page_number}"

    def __lt__(self, other):
        return self.comparison_attributes(self) < self.comparison_attributes(other)

    def __le__(self, other):
        return self.comparison_attributes(self) <= self.comparison_attributes(other)

    def __gt__(self, other):
        return self.comparison_attributes(self) > self.comparison_attributes(other)

    def __ge__(self, other):
        return self.comparison_attributes(self) >= self.comparison_attributes(other)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, RulingCitation):
            return self.year == other.year and self.volume == other.volume and self.page_number == other.page_number
        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

