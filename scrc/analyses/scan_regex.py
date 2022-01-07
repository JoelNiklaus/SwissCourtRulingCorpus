import pandas as pd
import numpy as np
from pprint import pprint

s = pd.Series()

def analyse_regex(paragraphs: list):
    global s
    listToAppend = []
    for item in paragraphs:
        if item in s.index:
            s[item] += 1
        else:
            listToAppend.append(item)
    count = [1] * len(listToAppend)
    s1 = pd.Series(count, listToAppend)
    s = s.append(s1)
    new_s = s[s > 10].sort_values(ascending=False)
    print(s.size)
    if new_s.size > 30:
        pprint(new_s)

def apply_regex(paragraph: str):
    return 0
    