from enum import Enum


class Split(Enum):
    ALL = "all"
    VALIDATION = "validation"
    TEST = "test"
    TRAIN = "train"
    SECRET_TEST = "secret_test"
