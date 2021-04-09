# Solving memory leak problem in pandas
# https://github.com/pandas-dev/pandas/issues/2659#issuecomment-12021083
import sys

import pandas as pd
from ctypes import cdll, CDLL

try:
    cdll.LoadLibrary("libc.so.6")
    libc = CDLL("libc.so.6")
    libc.malloc_trim(0)
except (OSError, AttributeError):
    libc = None

__old_del = getattr(pd.DataFrame, '__del__', None)


def __new_del(self):
    if __old_del:
        __old_del(self)
    libc.malloc_trim(0)


if libc:
    print('Applying monkeypatch for pd.DataFrame.__del__', file=sys.stderr)
    pd.DataFrame.__del__ = __new_del
else:
    print('Skipping monkeypatch for pd.DataFrame.__del__: libc or malloc_trim() not found', file=sys.stderr)
