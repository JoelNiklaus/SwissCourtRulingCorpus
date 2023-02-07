from operator import attrgetter

from scrc.data_classes.citation import Citation
from scrc.data_classes.law import Law
from scrc.utils.law_util_singleton import LawUtilSingleton

law_util = LawUtilSingleton()


class LawCitation(Citation):
    cit_as_found: str  # how the citations was found in text
    article: str  # not an int because it can also be sth like 7a
    paragraph: int = None  # optional
    numeral: int = None  # optional
    law: Law
    # compare first by law, then by article, then by paragraph and finally by numeral
    comparison_attributes = attrgetter("law", "article", "paragraph", "numeral")

    def __init__(self, citation_str, law_abbrs):
        """ 
            law_abbrs may contain data that is not properly cleaned and start with a space. Therefore
            it is necessary to strip the fields you want to extract.
        """
        self.cit_as_found = citation_str
        if citation_str.startswith("Art") or 'Abs' in citation_str or 'Ziff' in citation_str:
            self.article_str = "Art."
            self.paragraph_str = "Abs."
            self.numeral_str = "Ziff."
        else:
            self.article_str = "art."
            self.paragraph_str = "al."
            if 'n.' in citation_str:
                self.numeral_str = 'n.'
            if 'cpv.' in citation_str:
                self.numeral_str = 'cpv.'

        # sometimes there is a difference between § and the article_str, but we disregard it for simplicity
        citation_str = citation_str.replace("§", self.article_str)

        # insert dot after art if not there yeat
        if citation_str.startswith(self.article_str[:-1]) and citation_str[3] != ".":
            citation_str = citation_str[:3] + "." + citation_str[3:]  # insert the dot

        # make sure citation starts with art.
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

        law = law_abbrs[(law_abbrs.abbreviation.str.strip() == abbreviation)]  # cannot differ French and Italian
        if len(law.index) == 0:
            # only include citations that we can find in our corpus
            raise ValueError(f"The abbreviation ({abbreviation}) cannot be found.")
        sr_number = law.iloc[0].sr_number  # sr_number is for all languages the same

        self.law = Law(sr_number, law_abbrs)
        # TODO we could extend this to also extract optional paragraphs or numerals

    def __str__(self):
        str_repr = f"{self.article_str} {self.article}"
        if self.paragraph:
            str_repr += f" {self.paragraph_str} {self.paragraph}"
        if self.numeral:
            str_repr += f" {self.numeral_str} {self.numeral}"
        str_repr += f" {self.law}"
        return str_repr

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
        if isinstance(other, LawCitation):
            return self.article == other.article and self.law == other.law
        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))
