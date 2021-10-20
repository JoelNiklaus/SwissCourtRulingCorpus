from enum import Enum


class Judgment(Enum):
    APPROVAL = 'approval'
    PARTIAL_APPROVAL = 'partial_approval'
    DISMISSAL = 'dismissal'
    PARTIAL_DISMISSAL = 'partial_dismissal'
    INADMISSIBLE = 'inadmissible'
    WRITE_OFF = 'write_off'
    UNIFICATION = 'unification'
    # OTHER = 'other'
