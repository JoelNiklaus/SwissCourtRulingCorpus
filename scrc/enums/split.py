from enum import Enum


class Split(Enum):
    ALL = "all"
    VAL = "val"
    TEST = "test"
    TRAIN = "train"
    SECRET_TEST = "secret_test"