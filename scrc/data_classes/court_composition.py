import json
from dataclasses import dataclass, field
from typing import List
from scrc.data_classes.court_person import CourtPerson


@dataclass
class CourtComposition:
    president: CourtPerson = None
    judges: List[CourtPerson] = field(default_factory=list)
    clerks: List[CourtPerson] = field(default_factory=list)
    
    
    def toJSON(self):
        dict_representation = {
            'president': self.president.toJSON() if self.president else None,
            'judges': [
                judge.toJSON() for judge in self.judges
            ],
            'clerks': [
                clerk.toJSON() for clerk in self.clerks
            ]
        }
        return json.dumps(dict_representation, 
            sort_keys=True, indent=4)