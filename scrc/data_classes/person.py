from dataclasses import dataclass
from typing import Optional

from scrc.enums.gender import Gender


@dataclass
class Person:
    name: str
    gender: Gender = None
