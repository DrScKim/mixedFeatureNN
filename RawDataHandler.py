import time
import logging
import csv
import pandas as pd
import numpy as np
from enum import Enum

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format='%(message)s')

query = 0
id = 1
query_item_rank = 2
price = 3
impressionCount = 4
clickCount = 5
category = 6
brand = 7
title = 8

ID_COLUMN_NAMES = 999
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
        raw_data[ID_COLUMN_NAMES] = [Column.id]
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
    pass

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
    return df

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



def __featureMixer(numFeatDim, cateFeatDim, txtFeatDim):
    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ]
    )
    # query,  id, query_item_rank,    price,  impressionCount,    clickCount, category,   brand,  title

    num_feat_idx = [x for x in range(1,1+numFeatDim)]
    cate_feat_idx = [x for x in range(1+numFeatDim, 1+numFeatDim+cateFeatDim)]
    txt_feat_idx = [x for x in range(1+numFeatDim+cateFeatDim, 1+numFeatDim+cateFeatDim+txtFeatDim)]

    mixedFeature = ColumnTransformer(
        transformers=[
            ('numFeat', numerical_transformer, num_feat_idx),
            ('cateFeat', categorical_transformer, cate_feat_idx),
            ('txtFeat', 'passthrough', txt_feat_idx)
        ],
        remainder='drop',
        transformer_weights={
            'numFeat': 1.0,
            'cateFeat': 10.0,
            'txtFeat': 100.0,
        }
    )
    return mixedFeature



def mixingFeaturePipeline(raw_data):
    from sklearn.pipeline import Pipeline
    outputs = list()
    i = 0
    NNeighbour = 5 + 1
    candidateResult = dict()
    for query in raw_data[Q]:
        logging.info("{}) classified".format(query))
        #raw_dt[Q][query][CATE_MATRIX] = np.zeros(shape=raw_)
        numFeat = genNumericMatrix(raw_data, query)
        cateFeat = genCateMatrix(raw_data, query)
        txtFeat = genTitleMatrix(raw_data, query)
        id_list = pd.DataFrame(raw_data[Q][query][MATRIX_ID_TABLE])

        numFeatDim = numFeat.shape[1]
        cateFeatDim = cateFeat.shape[1]
        txtFeatDim = txtFeat.shape[1]

        feature = pd.concat([id_list, numFeat, cateFeat, txtFeat], axis=1, keys=['id','num','cate','text'])
        n_sample = min(feature.shape[0], NNeighbour)
        mixedFeature = __featureMixer(numFeatDim, cateFeatDim, txtFeatDim)
        clf = Pipeline(
            steps=[
                ('featureMixing', mixedFeature),
                ('clustering', NearestNeighbors(n_neighbors=n_sample, ))
            ]
        )
        result = clf.fit(feature)

        NN = result.named_steps['clustering']

        recommend(raw_data, query, NN.kneighbors(mixedFeature.fit_transform(feature)), feature, candidateResult, True)
    trace(raw_data, candidateResult)
    return candidateResult

def recommend(raw_data, query, NN, feature, candidate, isLogging=False):
    neighbour_list = NN[1]
    data = raw_data[Q][query]
    candidate[query] = dict()
    for feat in neighbour_list:
        o_id = feature.iloc[feat[0]]['id'].values[0]
        origin_rank = data[ITEM][o_id][Column.query_item_rank]
        if origin_rank > 10:
            continue
        original_price = data[ITEM][o_id][Column.price]

        candidate[query][o_id] = list()
        for neighbour in feat[1:]:
            c_id = feature.iloc[neighbour]['id'].values[0]
            candidate[query][o_id].append(c_id)

def trace(raw_data, candidates):
    for query in candidates:
        traceInfo = "query: {} \n".format(query)
        traceInfo += "=====================================================================================\n"
        for o_id in candidates[query]:
            traceInfo += "{:>12})[{:>20}:{:>8}]{:>7}:{:>35}\t{}\n".format (
                "original",
                raw_data[Q][query][ITEM][o_id][Column.category],
                raw_data[Q][query][ITEM][o_id][Column.brand],
                raw_data[Q][query][ITEM][o_id][Column.query_item_rank],
                raw_data[Q][query][ITEM][o_id][Column.title],
                raw_data[Q][query][ITEM][o_id][Column.price]
            )
            traceInfo += "=====================================================================================\n"
            for c_id in candidates[query][o_id]:
                traceInfo += "{:>12})[{:>20}:{:>8}]{:>7}:{:>35}\t{}\n".format(
                    "recommended",
                    raw_data[Q][query][ITEM][c_id][Column.category],
                    raw_data[Q][query][ITEM][c_id][Column.brand],
                    raw_data[Q][query][ITEM][c_id][Column.query_item_rank],
                    raw_data[Q][query][ITEM][c_id][Column.title],
                    raw_data[Q][query][ITEM][c_id][Column.price]
                )
            traceInfo += "=====================================================================================\n"
        logging.info(traceInfo)


if __name__ == "__main__":
    path = "./rawData.csv"
    raw_data = readData(path)
    candidates = mixingFeaturePipeline(raw_data)
    trace(raw_data, candidates)