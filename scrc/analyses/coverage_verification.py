
from sqlite3 import Connection
from symbol import stmt
from scrc.preprocessors.abstract_preprocessor import AbstractPreprocessor
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config
from sqlalchemy.sql.schema import Table, MetaData
from sqlalchemy.sql import select, func




class CoverageVerification(AbstractPreprocessor):
    "Ouputs docx files of court rulings with highlighted paragraphs of each section recognized aswell as the sentence which indicated the ruling outcome"
    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.decision = []
        self.logger_info = {
            'start': 'Started coverage verification',
            'finished': 'Finished coverage verification',
            'start_spider': 'Started coverage verification for spider',
            'finish_spider': 'Finished coverage verification for spider',
        }
    
    def start(self):
        self.get_random_decisions()
        
        
    def get_random_decisions(self):
        engine = self.get_engine(self.db_scrc)
        with engine.connect() as conn:
            d = Table('decision', MetaData(), autoload_with = engine)
            stmt = select(d.columns.decision_id).order_by(func.random()).limit(50)
            result = conn.execute(stmt)
            for res in result:
                self.get_sections(res, d, conn, engine)
                
    def get_sections(self, decision_id: str, d_table: Table, conn: Connection, engine):
        d = Table('decision', MetaData(), autoload_with = engine)
        result = conn.execute(stmt)


        
        
    
        
if __name__ == '__main__':
    config = get_config()

    coverage_verification = CoverageVerification(config)
    coverage_verification.start()