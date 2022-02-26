from scrc.data_classes.citation import Citation


class LawCitation(Citation):
    article: str
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
        if not citation_str.startswith(self.article_str):
            raise ValueError(f"The Citation String ({citation_str}) does not start with {self.article_str}.")
        parts = citation_str.split(" ")
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
