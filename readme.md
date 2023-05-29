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
These similarities will be the features used for classification. The keys 
that were compared were title, artists, isrcs and contributors. To compute the similarity 
between each string we used fuzz.WRatio, which is a composite approach of several weighted
comparisons.

### train_simple_threshold.py

This approach only performs a statistical analysis of the features, considering that
the higher the similarity the higher the probability that the entries being compared
are in fact the same song. Basically, the approach is that for each entry we get the mean of
the features (the 4 scores, the comparison between titles, artists, isrcs and contributors)
and then we compute the mean and std for the score of the valid comparisons and 
the same for the invalid comparisons.

As shown in the script, we get the following results:

Valid samples -> mean score: 81.320, std score: 15.178

Invalid samples -> mean score: 48.144, std score: 14.687

A reasonable approximation would be setting the threshold to 64.0, meaning that if the mean 
of the comparison is higher or equal than 64.0, we assume that that comparison is valid.

### train_NN.py

For this approach we use machine learning, taking advantage of the tools to perform model 
selection given by keras.

### train_random_forests.py

We train a classifier using sklearn and random forests.