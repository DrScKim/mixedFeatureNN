# mixedFeatureNN

This project contains how to find Nearest neighbour by mixing Heterogeneous features
such as **Numerical Feature, Categorical Feature, Textual Feature**


## Data scheme

**rawData.csv** contains raw data crawled from a E-Commerce site in S.Korea
textual data is written in Korean

Data scheme is like below

**query,  id, query_item_rank,    price,  impressionCount,    clickCount, category,   brand,  title**

* impressionCount and clickCount is written as Random number

this Data is indexed by query

## Description of this project

* This project shows simple Relevance feedback based Item recommendation.
* Relevance feedback:  https://en.wikipedia.org/wiki/Relevance_feedback
* This script response an higher ranked item A and items which are lower ranked and substitute for A  
  (potentially sellable)

## Dependencies
* Python 3.5+
* Numpy 1.16.1+
* pandas 0.24.1+
* Scikit-learn 0.20.2+
* scipy 1.2.1+
```
you can use requirements file as below
pip install -r requirements.txt
```


## Description of functions

RawDataHandler.py

function
* genNumericMatrix:
```python

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
```
* genCateMatrix:

* genTitleMatrix: query indexed document

* __featureMixer:

* mixingFeaturePipeline:

* recommend:
