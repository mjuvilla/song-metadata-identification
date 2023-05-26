# song-metadata-identification

This is a small project to test different approaches to manage song 
identification using its metadata.

The input data consists on two files. The first is groundtruth.csv, which 
for each line contains 
three columns, q_source_id, m_source_id and tag. "q_source_id" and 
"m_source_id" are two song ids to be compared and "tag" tells us if the 
comparison is valid (the two songs are equal) or invalid.

The second file is "db.db", a database that relates a song id (srid) with its
metadata. In this project we will only use title, artists, isrcs and 
contributors.

## Scripts
### data.py
This file contains two classes:

GroundTruth: loads and splits into train, validation and test datasets from
groundtruth.csv. These datasets are generated first by shuffling the whole groundtruth 
and then slicing it so train gets a 60% of the samples, validation 20% and test the 
remaining 20%. The train dataset will be used to train the classifier, the validation 
test will be used to test how the classifier is performing every time I make a change 
or tune some parameter and the test dataset will only be used at the end of the process 
to assess the performance of the system with samples that we have not used to train or 
fine-tune. Note: I will be using the same terminology as provided in the ground truth, 
namely, "Valid" meaning a true comparison (both songs are the same) and "Invalid" 
(the songs are different).

Database: loads the db.db file and provides a function that requires a sr_id and returns
the information contained in the db for that id, namely title, artists, isrcs and 
contributors.

### features.py
This file contains two functions:

data_to_features: given a dataset (previously acquired using the GroundTruth class) 
extracts for each entry the features we will be using for the classification.

song_similarity: given two recording entries computes the similarities between them. 
These similarities will be the features used for classification. As suggested, the keys 
that were compared were title, artists, isrcs and contributors. To compute the similarity 
between each string we used fuzz.WRatio, which is a composite approach of several weighted
comparisons.


measurements.py
This file contains two functions, one to compute the accuracy of the predictions and 
another to compute precision, recall and F-score, which are usual measurements for 
machine learning applications. These class is only used in the simple classifier since i