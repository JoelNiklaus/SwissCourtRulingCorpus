from operator import index
import pandas as pd
import numpy as np
import re
from pprint import pprint

from pandas.core.frame import DataFrame


variations = [(r'^', r'$'), (r'^', r'\s\d\d$'), (r'^', r'.$'), (r'^', r'\s$'), (r'^', r'\s.$')]

columns = ['Keyword/Sentence', 'Total Count'] + variations

dataArray = pd.DataFrame(columns= columns)

def analyse_regex(paragraphs: list):
    global dataArray
    newRow = [1] + [0] * len(variations)
    listToAppend = []
    for item in paragraphs:
        if len(item) < 50:
            foundInstance = False
            for idx, element in enumerate(variations):
                indexList = dataArray['Keyword/Sentence'].tolist()
                pattern = apply_regex(item, element)
                instance = [i for i, item in enumerate(indexList) if re.search(pattern, item)]
                if instance:
                    if len(instance) > 1:
                        mergeRows(instance, indexList, idx, item)
                    else:
                        dataArray.at[instance[0], element] += 1
                        dataArray.at[instance[0], 'Total Count'] += 1
                    foundInstance = True
            if not foundInstance:
                listToAppend.append([item] + newRow)
    dfTemp = pd.DataFrame(listToAppend, columns=columns)
    dataArray = pd.concat([dataArray, dfTemp], ignore_index=True)
    dataArray = dataArray.infer_objects()
    
def apply_regex(paragraph: str, pattern):
    sentence = re.escape(paragraph)
    wildcard = pattern[0] + sentence + pattern[1]
    return wildcard

def iterate_variations():
    for idx, element in enumerate(variations):
        print('t')

def mergeRows(indexes: list, indexList: list, patternPosition, word):
    global dataArray
    mergingRow = {}
    selectedRows: DataFrame = dataArray.iloc[indexes]
    selectedRows = selectedRows.select_dtypes(include=['int'])
    subset = selectedRows.sum(axis = 0, skipna =True)
    for i in selectedRows.columns:
        mergingRow[i] = subset[i]
        if i == variations[patternPosition]:
            mergingRow[i] += len(indexes)
    mergingRow['Total Count'] += 1
    mergingRow['Keyword/Sentence'] = indexList[indexes[0]]
    dataArray.drop(indexes, inplace=True)
    dataArray.reset_index(drop=True, inplace=True)
    dataArray = dataArray.append(mergingRow, ignore_index=True)
    print(dataArray, "after drop")
    