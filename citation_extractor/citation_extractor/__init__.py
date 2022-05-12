import json

import regex


def extract_citations(search_string: str, file_path: str, language: str):
    """Extracts the citation from a string

    Args:
        search_string (str): The text which the citations are extracted from
        file_path (str): The path to the file containing the citation regexes
        language (str): Any of the supported language abbreviations
    """
    def get_combined_regexes(regex_dict, language):
        return regex.compile("|".join([entry["regex"] for entry in regex_dict[language] if entry["regex"]]))

    citation_regexes = json.loads(
        (file_path).read_text())
    rulings = []
    laws = []

    for match in regex.findall(get_combined_regexes(citation_regexes['ruling']['BGE'], language), search_string):
        rulings.append(match.group(0))

    for match in regex.findall(get_combined_regexes(citation_regexes['ruling']['Bger'], language), search_string):
        rulings.append(match.group(0))

    for match in regex.findall(get_combined_regexes(citation_regexes['law'], language), search_string):
        laws.append(match.group(0))

    return {"rulings": rulings, "laws": laws}
