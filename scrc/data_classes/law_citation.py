from scrc.data_classes.citation import Citation


class LawCitation(Citation):
    article: str  # not an int because it can also be sth like 7a
    paragraph: int = None  # optional
    numeral: int = None  # optional
    abbreviation: str

    def __init__(self, citation_str, language="de"):
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
        self.abbreviation = parts[-1]  # should be the last part
        # TODO we can extend this to also extract optional paragraphs or numerals

    def __str__(self):
        str_repr = f"{self.article_str} {self.article}"
        if self.paragraph:
            str_repr += f" {self.paragraph_str} {self.paragraph}"
        if self.numeral:
            str_repr += f" {self.numeral_str} {self.numeral}"
        str_repr += f" {self.abbreviation}"
        return str_repr

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, LawCitation):
            return self.article == other.article and self.abbreviation == other.abbreviation
        return False

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))
