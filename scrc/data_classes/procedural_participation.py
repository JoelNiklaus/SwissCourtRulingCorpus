from dataclasses import dataclass, field
from typing import List
import json

from scrc.data_classes.proceedings_party import ProceedingsParty


@dataclass
class ProceduralParticipation:

    plaintiffs: List[ProceedingsParty] = field(default_factory=list)# BeschwerdeführerIn / Kläger
    defendants: List[ProceedingsParty] = field(default_factory=list) # Beklagter

    def toJSON(self):
        dict_representation = {
            'plaintiffs': [{
                'legal_type': party.legal_type.value,
                'legal_counsel': [{
                    'name': counsel.name,
                    'gender': counsel.gender,
                    'legal_type': counsel.legal_type.value
                } for counsel in party.legal_counsel]
            } for party in self.plaintiffs],
            
            'defendants': [{
                'legal_type': party.legal_type.value,
                'legal_counsel': [{
                    'name': counsel.name,
                    'gender': counsel.gender,
                    'legal_type': counsel.legal_type.value
                } for counsel in party.legal_counsel]
            } for party in self.defendants],
        }
        return json.dumps(dict_representation, 
            sort_keys=True, indent=4)
