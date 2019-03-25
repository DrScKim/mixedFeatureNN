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
convert raw data of numeric features to data frame format
* genCateMatrix:
convert raw data of categorical features to data frame format
* genTitleMatrix: query indexed document
convert raw data of title data( textual data ) to data frame format
* __featureMixer:
Mix all features by using Pipeline and ColumnTransformer in Scikit-learn
if you want to use some features, you can select index of those feature you want to select
this logic only use selected feature containing in "xxx_feat_idx"
```python
def __featureMixer(numFeatDim, cateFeatDim, txtFeatDim, num_feat_idx=None, cate_feat_idx=None, txt_feat_idx=None)
```

    * numFeatDim:
    * cateFeatDim:
    * txtFeatDim:
    * num_feat_idx:
    * cate_feat_idx:
    * txt_feat_idx:
    
  
* mixingFeaturePipeline:

* recommend:
