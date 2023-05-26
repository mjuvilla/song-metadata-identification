from fuzzywuzzy import fuzz
import tensorflow as tf

tf.random.set_seed(0)

def song_similarity(sr1, sr2):
    similarity = {}
    for key in sr1.keys():
        similarity[key] = fuzz.WRatio(sr1[key], sr2[key])

    return similarity


def data_to_features(dataset, data_handler):
    x = []
    y = []

    for sr1_key, sr2_key, target in dataset:
        sr1 = data_handler.get_sr_by_srid(sr1_key)
        sr2 = data_handler.get_sr_by_srid(sr2_key)

        similarity = song_similarity(sr1, sr2)

        # using data.columns so the features are always in the same order instead of doing similarity.keys()
        x.append([similarity[key] for key in data_handler.columns])
        y.append(1 if target == "valid" else 0)

    return tf.convert_to_tensor(x), tf.convert_to_tensor(y)
