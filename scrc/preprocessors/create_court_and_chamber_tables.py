import json
import pandas as pd
from sqlalchemy import MetaData, Table
from root import ROOT_DIR

from scrc.preprocessors.abstract_preprocessor import AbstractPreprocessor
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config

class CreateCourtAndChamberTables(AbstractPreprocessor):
    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)
    

    def start(self):
        self.logger.info('Checking for new courts and chambers')
        courts_chambers_file = json.loads((ROOT_DIR / "legal_info/court_chambers.json").read_text())
        
        self.existing_cantons = self.get_cantons()
        
        self.existing_courts = self.get_existing_courts() # Get current courts
        self.add_courts(courts_chambers_file) # Add courts not present yet
        self.existing_courts = self.get_existing_courts() # Reload so new courts are included
    
        self.existing_chambers = self.get_existing_chambers() # Get current chambers
        self.spiders = self.get_spiders() # Get the spiders
        self.add_chambers(courts_chambers_file) # Add chambers not present yet
    
    def get_cantons(self)-> pd.DataFrame:
        return list(self.select(self.get_engine(self.db_scrc), 'canton'))[0]
    
    def get_existing_courts(self):
        return list(self.select(self.get_engine(self.db_scrc), 'court'))[0]
    
    def get_existing_chambers(self):
        return list(self.select(self.get_engine(self.db_scrc), 'chamber'))[0]
    
    def get_spiders(self):
        return list(self.select(self.get_engine(self.db_scrc), 'spider'))[0]  
            
    def add_courts(self, courts_chambers_file): 
        with self.get_engine(self.db_scrc).connect() as conn:
            t_court = Table('court', MetaData(), autoload_with=conn)
            for canton in courts_chambers_file:
                canton_id = int(self.existing_cantons.loc[self.existing_cantons['short_code'] == canton]['canton_id'].iloc[0])
                for court in courts_chambers_file[canton]['gerichte']:
                    if not court in list(self.existing_courts['court_string']):
                        # Add court to the database
                        court_dict = {'canton_id': canton_id, "court_string": court}
                        stmt = t_court.insert().values([court_dict])
                        conn.execute(stmt)
                        self.logger.info(f"Added new court {court}")
                        
    def add_chambers(self, courts_chambers_file): 
        with self.get_engine(self.db_scrc).connect() as conn:
            t_chamber = Table('chamber', MetaData(), autoload_with=conn)
            for canton in courts_chambers_file:
                for court in courts_chambers_file[canton]['gerichte']:
                    court_id = int(self.existing_courts.loc[self.existing_courts['court_string'] == court]['court_id'].iloc[0])
                    for chamber in courts_chambers_file[canton]['gerichte'][court]['kammern']:
                        if not chamber in list(self.existing_chambers['chamber_string']):
                            # Add chamber to the database
                            spider_name = courts_chambers_file[canton]['gerichte'][court]['kammern'][chamber].get('spider')
                            spider_id = int(self.spiders.loc[self.spiders['name'] == spider_name]['spider_id'].iloc[0])
                            
                            chamber_dict = {"court_id": court_id, "spider_id": spider_id, "chamber_string": chamber}
                            stmt = t_chamber.insert().values([chamber_dict])
                            conn.execute(stmt)
                            self.logger.info(f"Added new chamber {chamber}")
    
if __name__ == '__main__':
    config = get_config()

    create_court_and_chamber_tables = CreateCourtAndChamberTables(config)
    create_court_and_chamber_tables.start()
