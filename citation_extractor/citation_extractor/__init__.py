import json
from pathlib import Path

import re


def extract_citations(search_string: str, language: str, filepath: str = ''):
    """Extracts the citation from a string

    Args:
        search_string (str): The text which the citations are extracted from
        file_path (str): The path to the file containing the citation regexes
        language (str): Any of the supported language abbreviations
    """
    def get_combined_regexes(regex_dict, language):
        try:
            comp_regex = re.compile("|".join([entry["regex"] for entry in regex_dict[language] if entry["regex"]]))
            return comp_regex
        except:
            raise NotImplementedError("Not implemented the requested regexes")
        
    def delete_duplicates(citation_list):
        """ 
            Multiple Regex can find the same shortened version of citations, so they will be excluded. 
            For example Art. 10 StGB could be caught by multiple regexes, so they get filtered out.
        """
        return_set = list()
        return_set_texts = list()
        for item in citation_list:
            if item['text'] in return_set_texts:
                continue
            return_set.append(item)
            return_set_texts.append(item['text'])
            
    def clean_citation_text(citation_text: str) -> str:
        """ Strips the citation and fixes the problem that the law can be mentioned twice in the match. (Art. 147 Abs. 1 StGB   StGB) """
        citation_text = citation_text.strip()
        citation_text = re.sub(r'\s+', ' ', citation_text)
        citation_text_parts = citation_text.split()
        if len(citation_text_parts) >= 2 and citation_text_parts[-1] == citation_text_parts[-2]:
            citation_text = ' '.join(citation_text_parts[:-1])
        else:
            citation_text = ' '.join(citation_text_parts)
        return citation_text
            

    if filepath is None or filepath == '':
        filepath: Path = Path(__file__).parent / 'citation_regexes.json'
    else:
        filepath: Path = Path(filepath)

    citation_regexes = json.loads(filepath.read_text())
    rulings = []
    laws = []
    try:
        for match in re.findall(get_combined_regexes(citation_regexes['ruling']['BGE'], language), str(search_string)):
            citation = {"type": "bge", "text": " ".join(match).strip()}
            rulings.append(citation)

        for match in re.findall(get_combined_regexes(citation_regexes['ruling']['Bger'], language), str(search_string)):
            citation = {"type": "bger", "text": " ".join(match).strip()}
            rulings.append(citation)

        for match in re.findall(get_combined_regexes(citation_regexes['law'], language), str(search_string)):
            citation = {"text": " ".join(match)}
            citation['text'] = clean_citation_text(citation['text'])
            laws.append(citation)

        
    except Exception as e:
        print(f'Error while extracting citations: {e}')
    
    for citation_list in [rulings, laws]:
        delete_duplicates(citation_list)
        
    return {"rulings": rulings, "laws": laws}
    
