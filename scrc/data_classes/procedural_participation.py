from dataclasses import dataclass
from typing import List

from scrc.data_classes.proceedings_party import ProceedingsParty


@dataclass
class ProceduralParticipation:
    plaintiffs: List[ProceedingsParty]  # BeschwerdeführerIn / Kläger
    defendants: List[ProceedingsParty]  # Beklagter
