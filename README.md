## timeseriesVectorDBPatterMatcher

This moduel generate embedding from a large number of time series, save them to pinecone vector DB then match a new time series using vector similarity against the period in the database
## Data pack generation


To generate the datapack that can be used to populate the pinecone vector database, run the followings

python katslib/opkatsDataUtil.py

run this notebook (kats_pattern_vectordb.ipynb) to see the pattern matching outcome
>>>>>>> origin/master
