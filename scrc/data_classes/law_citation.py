from scrc.data_classes.citation import Citation
from scrc.data_classes.law import Law
from scrc.utils.law_util_singleton import LawUtilSingleton

law_util = LawUtilSingleton()


class LawCitation(Citation):
    article: str  # not an int because it can also be sth like 7a
    paragraph: int = None  # optional
    numeral: int = None  # optional
    law: Law

    def __init__(self, citation_str, law_abbrs, language="de"):
        self.language = language
        if language == "de":
            self.article_str = "Art."
            self.paragraph_str = "Abs."
            self.numeral_str = "Ziff."
        elif language == "fr":
            self.article_str = "art."
            self.paragraph_str = "al."
            self.numeral_str = "n."
        elif language == "it":
            self.article_str = "art."
            self.paragraph_str = "al."
            self.numeral_str = "cpv."
        # sometimes there is a difference between § and the article_str, but we disregard it for simplicity
        citation_str = citation_str.replace("§", self.article_str)
        # quick hacky fix
        if citation_str.startswith(self.article_str[:-1]) and citation_str[3] != ".":
            citation_str = citation_str[:3] + "." + citation_str[3:]  # insert the dot
        if not citation_str.lower().startswith(self.article_str.lower()):
            raise ValueError(f"The Citation String ({citation_str}) does not start with {self.article_str}.")

        parts = citation_str.split(" ")
        if len(parts) == 2 and self.article_str in parts[0]:
            parts.insert(0, self.article_str)
            parts[1] = parts[1][len(self.article_str):]  # take everything after the article_str
        if len(parts) < 3:  # either an article number or the abbreviation is missing
            raise ValueError(f"The Citation String ({citation_str}) consists of less than 3 parts.")
        self.article = parts[1]  # should be the second part after "Art."
        abbreviation = parts[-1]  # should be the last part

        law = law_abbrs[(law_abbrs.abbreviation == abbreviation) & (law_abbrs.language == language)]
        if len(law.index) == 0:
            # only actually include citations that we can find in our corpus
            raise ValueError(f"The abbreviation ({abbreviation}) cannot be found.")
        assert len(law.index) == 1
        sr_number = law.iloc[0].sr_number
        abbreviations = law_abbrs[law_abbrs.sr_number == sr_number]
        abbreviations = abbreviations[["language", "abbreviation"]].set_index("language").to_dict()['abbreviation']

        self.law = Law(sr_number, abbreviations)
        # TODO we could extend this to also extract optional paragraphs or numerals

    def __str__(self):
        str_repr = f"{self.article_str} {self.article}"
        if self.paragraph:
            str_repr += f" {self.paragraph_str} {self.paragraph}"
        if self.numeral:
            str_repr += f" {self.numeral_str} {self.numeral}"
        str_repr += f" {self.law}"
        return str_repr

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, LawCitation):
            return self.article == other.article and self.law == other.law
        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))
