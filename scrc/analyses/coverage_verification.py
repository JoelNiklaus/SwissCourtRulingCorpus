
from enum import Enum
from pathlib import Path
from sqlite3 import Connection
from scrc.preprocessors.abstract_preprocessor import AbstractPreprocessor
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config
from scrc.enums.section import Section
from docx import Document


class Color(Enum):
    HEADER = 7
    FACTS = 8
    CONSIDERATIONS = 10
    RULINGS = 16
    FOOTER = 3


class CoverageVerification(AbstractPreprocessor):
    """Ouputs docx files of court rulings with highlighted paragraphs of each section recognized aswell as the sentence which indicated the ruling outcome. 
    To run simply execute python -m scrc.analyses.coverage_verification and make sure coverage_verification.txt exists."""
    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.logger_info = {
            'start': 'Started coverage verification',
            'finished': 'Finished coverage verification',
            'start_spider': 'Started coverage verification for spider',
            'finish_spider': 'Finished coverage verification for spider',
        }
        self.processed_file_path = self.progress_dir / "coverage_verification.txt"

    
    def start(self):
        document = Document()
        self.init_engine(document)
        
        
    def init_engine(self, document):
        spider_list, message = self.compute_remaining_spiders(self.processed_file_path)
        self.logger.info(message)
        engine = self.get_engine(self.db_scrc)
        with engine.connect() as conn:
            for spider in spider_list:
                for x in range(0, 20):
                    self.get_random_decision(conn, document, spider, 0)
                document.save(self.get_path(spider))     

            
    def get_random_decision(self, conn: Connection, document, spider, try_count):
        result = conn.execute(self.random_decision_query(spider))
        for res in result:
            section_dict = self.get_sections(res, conn)
            if not self.valid_decision(section_dict['sections']) and try_count <= 5:
                try_count += 1
                self.get_random_decision(conn, document, spider, try_count)
            elif try_count > 5:
                return
            else: 
                self.append_to_doc(section_dict, document, conn)
                
    def get_sections(self, result: str, conn: Connection,):
        section_dict = {'sections': {}, 'id': result[0]}
        result = conn.execute(self.section_query(result[0]))
        for res in result:
            section_dict['sections'][Section(res.section_type_id).name] = res.section_text
        return section_dict
    
    def valid_decision(self, sections):
        count = 0
        for section in sections:
            if sections[section] != '':
                count += 1
        return count > 2
        
    def append_to_doc(self, section_dict: dict, document: Document, conn: Connection):
        document.add_paragraph(str(section_dict['id']))
        sorted_list = sorted(list(section_dict['sections'].items()), key=lambda x: Section[x[0]].value)
        ruling = self.get_ruling_outcome(str(section_dict['id']), conn)
        for element in sorted_list:
            p = document.add_paragraph()
            r = p.add_run(f"{element[0]}: {element[1]}").font
            r.highlight_color = Color[element[0]].value
            if element[0] == 'RULINGS' and ruling:
                p = document.add_paragraph()
                r = p.add_run(ruling.text).bold = True         
        return document 
    
    def get_ruling_outcome(self, decision_id: str, conn: Connection):
        return conn.execute(self.ruling_outcome_query(decision_id)).fetchone()
        
    
    def ruling_outcome_query(self, decision_id: str):
        return (f"SELECT * FROM judgment "
                f"INNER JOIN judgment_map ON judgment.judgment_id = judgment_map.judgment_id "
                f"INNER JOIN decision ON judgment_map.decision_id = decision.decision_id "
                f"WHERE decision.decision_id = '{decision_id}'")
    
    def append_paragraph(self, section_text: str, section_type, document: Document):
        p = document.add_paragraph()
        r = p.add_run(section_text).font     
        r.highlight_color = Color[section_type].value 
        return document

    def section_query(self, decision_id):
        return (f"SELECT * FROM section "
                f"WHERE section.decision_id = '{decision_id}' "
                f"AND section.section_type_id != 1"
                f"AND section.section_type_id != 3")
        
    def random_decision_query(self, spider: str):
        return (f"SELECT * FROM decision "
                f"INNER JOIN chamber ON decision.chamber_id = chamber.chamber_id "
                f"INNER JOIN spider ON chamber.spider_id = spider.spider_id "
                f"WHERE spider.name = '{spider}' "
                f"ORDER BY RANDOM() LIMIT 1")
        
    def get_path(self, spider: str):
        filepath = Path(
        f'data/verification/{spider}.docx')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        return filepath
        
        
if __name__ == '__main__':
    config = get_config()

    coverage_verification = CoverageVerification(config)
    coverage_verification.start()