from dataclasses import dataclass, field
from typing import List, Optional

from scrc.enums.gender import Gender
from scrc.enums.title import Title


@dataclass
class Person:
    name: str
    gender: Gender = None
    titles: List[Title] = field(default_factory=list)

