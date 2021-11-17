from enum import Enum


class CourtRole(Enum):
    PRESIDENT = 'president'
    VICE_PRESIDENT = 'vice president'
    JUDGE = 'judge'
    JUDGE_ASSESSOR = 'judge assessor'
    JUDGE_SUPPLEMENTARY = 'judge supplementary'
    JUDGE_REPORTER = 'judge reporter'
    DELEGATE_JUDGE = 'delegate judge'
    ASSESSOR = 'assessor'
    CLERK = 'clerk'
