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


## Description of functions

RawDataHandler.py

function
* genNumericMatrix:

* genCateMatrix:

* genTitleMatrix: query indexed document

* __featureMixer:

* mixingFeaturePipeline:

* recommend:
