from dataclasses import dataclass, field
from typing import List
import json

from scrc.data_classes.proceedings_party import ProceedingsParty


@dataclass
class ProceduralParticipation:
    plaintiffs: List[ProceedingsParty] = field(default_factory=list)# BeschwerdeführerIn / Kläger
    defendants: List[ProceedingsParty] = field(default_factory=list) # Beklagter

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)
