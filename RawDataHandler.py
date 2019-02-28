import time
import logging
import csv
import pandas as pd
import numpy as np
from enum import Enum

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

query = 0
id = 1
query_item_rank = 2
price = 3
impressionCount = 4
clickCount = 5
category = 6
brand = 7
title = 8


NUMERIC_COLUMN_NAMES = 1000
CATEGORICAL_COLUMN_NAMES = 1001
TEXTUAL_COLUMN_NAMES = 10002
CATE_MATRIX = 10003
MATRIX_ID_TABLE = 10004
TF_IDF_FEATURE = 10005
NUMERIC_MATRIX = 10006

Q = 101
ITEM = 102

class Column(Enum):
    query = 0
    id = 1
    query_item_rank = 2
    price = 3
    impressionCount = 4
    clickCount = 5
    category = 6
    brand = 7
    title = 8
'''
data scheme
query,  id, query_item_rank,    price,  impressionCount,    clickCount, category,   brand,  title
nike, 1, 1, 100, 1000,

'''


def readData(path):
    with open(path, 'r') as rfp:
        raw_data = dict()
        raw_data[Q] = dict()
        raw_data[NUMERIC_COLUMN_NAMES] = [Column.query_item_rank, Column.price,
                                         Column.impressionCount, Column.clickCount,]
        raw_data[CATEGORICAL_COLUMN_NAMES] = [Column.category, Column.brand,]
        raw_data[TEXTUAL_COLUMN_NAMES] = [Column.title, ]
        rcsv = csv.reader(rfp)
        next(rcsv)
        for row in rcsv:
            query = row[Column.query.value]
            id = int(row[Column.id.value])

            if query not in raw_data[Q]:
                raw_data[Q][query] = dict()
                raw_data[Q][query][ITEM] = dict()
            if id not in raw_data[Q][query][ITEM]:
                raw_data[Q][query][ITEM][id] = dict()
            for col in raw_data[NUMERIC_COLUMN_NAMES]:
                raw_data[Q][query][ITEM][id][col] = float(row[col.value])
            for col in raw_data[CATEGORICAL_COLUMN_NAMES]:
                raw_data[Q][query][ITEM][id][col] = row[col.value]
            for col in raw_data[TEXTUAL_COLUMN_NAMES]:
                raw_data[Q][query][ITEM][id][col] = row[col.value]

    return raw_data

def imputeNumericData(npMat):

def genNumericMatrix(raw_data, query):
    n_rows = len(raw_data[Q][query][ITEM].keys())
    n_cols = len(raw_data[NUMERIC_COLUMN_NAMES])
    l = np.ndarray(shape=(n_rows, n_cols))
    item_id_list = list()
    row = 0
    for id in raw_data[Q][query][ITEM]:
        item_id_list.append(id)
        data = [raw_data[Q][query][ITEM][id][x] for x in raw_data[NUMERIC_COLUMN_NAMES]]
        l[row] = np.array(data)
        row += 1
    raw_data[Q][query][NUMERIC_MATRIX] = l
    raw_data[Q][query][MATRIX_ID_TABLE] = item_id_list
    df = pd.DataFrame(raw_data[Q][query][NUMERIC_MATRIX])
    return

def genCateMatrix(raw_data, query):

    cateMat = list()
    for i in range(len(raw_data[Q][query][ITEM])):
        cateMat.append(list())

    for id in raw_data[Q][query][ITEM]:
        queryMatIdx = raw_data[Q][query][MATRIX_ID_TABLE].index(id)
        for idx, cateF in enumerate(raw_data[CATEGORICAL_COLUMN_NAMES]):
            cateMat[queryMatIdx].append(raw_data[Q][query][ITEM][id][cateF])
            #if raw_dt[Q][query][V][vi_id][cateF] == '':
            #    raw_dt[Q][query][V][vi_id][cateF] = np.nan
    df = pd.DataFrame(cateMat, dtype="category")
    return df

def tokenNormalizer(txt):
    return txt.replace("[", "").replace("]", "").replace("'", " ").replace("_", " ").replace("'", " ")

def genTitleMatrix(raw_data, query):
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    corpus = []
    #for query in raw_data[Q]:
    docs = list()
    for id in raw_data[Q][query][ITEM]:
        docs.append( raw_data[Q][query][ITEM][id][Column.title])
    tokens = " ".join(docs)
    tokens = tokenNormalizer(tokens)
    corpus.append(tokens)

    vectorizer = CountVectorizer()
    X = vectorizer.fit(corpus)

    titleList = list()
    raw_data[Q][query][TF_IDF_FEATURE] = [0] * len(raw_data[Q][query][ITEM])
    for id in raw_data[Q][query][ITEM]:
        titleList.append(tokenNormalizer(raw_data[Q][query][ITEM][id][Column.title]))
    raw_data[Q][query][TF_IDF_FEATURE] = X.transform(titleList)
    df = pd.DataFrame(raw_data[Q][query][TF_IDF_FEATURE].toarray())
    return df



def mixingFeaturePipeline(raw_data):
    from sklearn.pipeline import Pipeline
    outputs = list()
    i = 0

    for query in raw_data[Q]:
        logging.WARN("{}): classified")
        #raw_dt[Q][query][CATE_MATRIX] = np.zeros(shape=raw_)
        numFeat = genNumericMatrix(raw_data, query)
        cateFeat = genCateMatrix(raw_data, query)






def recommend():
    pass

if __name__ == "__main__":
    path = "./rawData.csv"
    readData(path)