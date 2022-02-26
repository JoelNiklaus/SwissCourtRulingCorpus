import json
from dataclasses import dataclass

from scrc.data_classes.person import Person
from scrc.enums.court_role import CourtRole
from scrc.enums.political_party import PoliticalParty


@dataclass
class CourtPerson(Person):
    court_role: CourtRole = None
    party: PoliticalParty = None
    
    
    def toJSON(self):
        intermediate_dict_representation = {
            'party': self.party.value if self.party else None,
            'court_role': self.court_role.value if self.court_role else None,
        }
        person_dict = super().toJSON()
        dict_representation = {**super().toJSON(), **intermediate_dict_representation}
        
        return json.dumps(dict_representation, 
            sort_keys=True, indent=4)
