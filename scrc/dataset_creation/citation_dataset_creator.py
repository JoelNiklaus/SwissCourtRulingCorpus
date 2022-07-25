import json

from abc import ABC
from tqdm import tqdm
from scrc.data_classes.ruling_citation import RulingCitation
from scrc.dataset_creation.dataset_creator import DatasetCreator
from pandarallel import pandarallel


class CitationDatasetCreator(ABC, DatasetCreator):
    """Abstract Base Class to unify methods for citatition_criticality_dataset_creator and
    doc2doc_ir_dataset_creator"""

    def __init__(self, config: dict):
        super().__init__(config)

        pandarallel.initialize(progress_bar=True)
        tqdm.pandas()

    def load_rulings(self):
        self.logger.info(f"Loading reference rulings")
        self.available_bges = set()  # use set instead of list for faster lookup
        for lang in self.languages:
            df = next(self.select(self.get_engine(self.db_scrc), lang,
                                  columns="file_number", where="spider = 'CH_BGE'", chunksize=self.real_chunksize))
            assert len(df.index) == len(df.file_number.unique())  # there should not be any duplicates
            self.available_bges.update(df.file_number.tolist())

    def get_citations(self, citations_as_string, lang):
        cits = []
        try:
            citations = list(eval(citations_as_string))
            for citation in citations:
                try:
                    cit = citation['text']
                    type = citation['name']
                    cit = ' '.join(cit.split())  # remove multiple whitespaces inside
                    if type == "ruling":
                        file_number = self.get_file_number(cit, lang)
                        type_cit = RulingCitation(file_number, lang)
                        cits.append(type_cit)
                    elif type == "law":
                        pass
                    else:
                        raise ValueError("type must be 'rulings' or 'law")
                except ValueError as ve:
                    self.logger.info("citation has invalid syntax")
                    self.logger.info(citation)
                    continue
        except ValueError as ve:
            self.logger.info("citations could not be extracted to dict")
            self.logger.info(citations_as_string)
        if cits:  # only return something if we actually have citations
            return cits

    def get_file_number(self, citation, lang):
        # make sure citation string starts with BGE as in availabel_bge
        if citation[0].isnumeric():
            citation = f"BGE {citation}"
        else:
            citation = citation.replace("ATF", "BGE")
            citation = citation.replace("DTF", "BGE")

        if str(citation) in self.available_bges:
            return str(citation)
        else:
            # find closest bge with smaller page_number
            ruling_cit = RulingCitation(citation, lang)
            year = ruling_cit.year
            volume = ruling_cit.volume
            page_number = ruling_cit.page_number
            new_page_number = -1
            for match in self.available_bges:
                if f"BGE {year} {volume}" in match:
                    bge = RulingCitation(match, lang)
                    if new_page_number < bge.page_number <= page_number:
                        new_page_number = bge.page_number
            return f"BGE {year} {volume} {new_page_number}"
