import json
from pathlib import Path

import regex


def extract_citations(search_string: str, language: str, filepath: str = ''):
    """Extracts the citation from a string

    Args:
        search_string (str): The text which the citations are extracted from
        file_path (str): The path to the file containing the citation regexes
        language (str): Any of the supported language abbreviations
    """
    def get_combined_regexes(regex_dict, language):
        try:
            comp_regex = regex.compile(
                "|".join([entry["regex"] for entry in regex_dict[language] if entry["regex"]]))
            return comp_regex
        except:
            raise NotImplementedError("Not implemented the requested regexes")

    if filepath is None or filepath == '':
        filepath: Path = Path(__file__).parent / 'citation_regexes.json'
    else:
        filepath: Path = Path(filepath)

    citation_regexes = json.loads(filepath.read_text())
    rulings = []
    laws = []

    for match in regex.findall(get_combined_regexes(citation_regexes['ruling']['BGE'], language), str(search_string)):
        rulings.append(" ".join(match).strip())

    for match in regex.findall(get_combined_regexes(citation_regexes['ruling']['Bger'], language), str(search_string)):
        rulings.append(" ".join(match).strip())

    for match in regex.findall(get_combined_regexes(citation_regexes['law'], language), str(search_string)):
        laws.append(" ".join(match).strip())

    return {"rulings": rulings, "laws": laws}
