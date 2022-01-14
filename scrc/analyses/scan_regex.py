from operator import index
import pandas as pd
import numpy as np
import re
from pprint import pprint

from pandas.core.frame import DataFrame


variations = [(r'^', r'$'), (r'^', r'\s\d\d$'), (r'^', r'.$'), (r'^', r'\s$'), (r'^', r'\s.$')]

columns = ['keyword', 'Total Count'] + variations

dataArray = pd.DataFrame(columns= columns)

def analyse_regex(paragraphs: list):
    global dataArray
    newRow = [1] + [0] * len(variations)
    listToAppend = []
    for item in paragraphs:
        if len(item) < 45:
            foundInstance = iterate_variations(item)
            if not foundInstance:
                listToAppend.append([item] + newRow)
    dfTemp = pd.DataFrame(listToAppend, columns=columns)
    dataArray = pd.concat([dataArray, dfTemp], ignore_index=True)
    dataArray = dataArray.infer_objects()
    
def apply_regex(paragraph: str, pattern):
    sentence = re.escape(paragraph)
    wildcard = pattern[0] + sentence + pattern[1]
    return wildcard

def iterate_variations(item: str):
    foundInstance = False
    for idx, element in enumerate(variations):
        indexList = dataArray['keyword'].tolist()
        pattern = apply_regex(item, element)
        instance = [i for i, item in enumerate(indexList) if re.search(pattern, item)]
        if instance:
            foundInstance = True
            if len(instance) > 1:
                searchTable(instance, indexList, idx, foundInstance)
            else:
                dataArray.at[instance[0], element] += 1
                dataArray.at[instance[0], 'Total Count'] += 1
    return foundInstance


def searchTable(indexes: list, indexList: list, patternPosition: int, alreadyFound: bool):
    global dataArray
    mergingRow = {}
    output = [indexList[index] for index in indexes]
    lcs = commonSubstring(output)
    mergingRow['keyword'] = lcs
    if lcs in dataArray['keyword'].tolist():
        indexes += dataArray.index[dataArray['keyword'] == lcs].tolist()
    selectedRows: DataFrame = dataArray.iloc[indexes]
    selectedRows = selectedRows.select_dtypes(include=['int'])
    subset = selectedRows.sum(axis = 0, skipna =True)
    for i in selectedRows.columns:
        mergingRow[i] = subset[i]
        if i == variations[patternPosition]:
            mergingRow[i] += len(indexes)
    if not alreadyFound:
        mergingRow['Total Count'] += 1
    dataArray.drop(indexes, inplace=True)
    dataArray.reset_index(drop=True, inplace=True)
    dataArray = dataArray.append(mergingRow, ignore_index=True)
    # print(dataArray.sort_values(by=['Total Count'], ascending=False), 'after Drop')
    
    
def commonSubstring(arr):
    n = len(arr)
    s = arr[0]
    l = len(s)
    res = ""
    for i in range(l):
        for j in range(i + 1, l + 1):
            stem = s[i:j]
            k = 1
            for k in range(1, n):
                if stem not in arr[k]:
                    break
            if (k + 1 == n and len(res) < len(stem)):
                res = stem
    return res
    