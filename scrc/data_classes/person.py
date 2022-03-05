import json
from dataclasses import dataclass, field
from typing import List, Optional

from scrc.enums.gender import Gender
from scrc.enums.title import Title


@dataclass
class Person:
    name: str
    gender: Gender = None
    titles: List[Title] = field(default_factory=list)

    def toJSON(self):
        dict_representation = {
            'name': self.name,
            'gender': self.gender.value if self.gender else None,
            'titles': [
                title.value for title in self.titles if title
            ]
        }
        
        return json.dumps(dict_representation, 
            sort_keys=True, indent=4)
