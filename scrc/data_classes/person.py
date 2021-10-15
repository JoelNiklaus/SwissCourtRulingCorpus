from dataclasses import dataclass

from scrc.enums.gender import Gender


@dataclass
class Person:
    name: str
    gender: Gender
