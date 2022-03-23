from enum import Enum


class Judgment(Enum):
    APPROVAL = 1
    PARTIAL_APPROVAL = 4
    DISMISSAL = 2
    PARTIAL_DISMISSAL = 5
    INADMISSIBLE = 3
    WRITE_OFF = 7
    UNIFICATION = 6
    # OTHER = 'other'
