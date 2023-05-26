import sqlite3
import os
import numpy as np
import random
import csv

seed = 0
random.seed(seed)
np.random.seed(seed)


def read_csv(groundtruth_path):
    data = []
    with open(groundtruth_path, "r") as infile:
        reader = csv.reader(infile, delimiter=',', quotechar='|')
        next(reader, None)  # skip the headers
        for row in reader:
            data.append(row)
    return data


class GroundTruth:
    def __init__(self, groundtruth_path):
        self.ground_truth = None
        self.train = None
        self.validate = None
        self.test = None
        self.get_ground_truth(groundtruth_path)

    def get_ground_truth(self, groundtruth_path):
        if not os.path.exists(groundtruth_path):
            raise FileNotFoundError

        self.ground_truth = read_csv(groundtruth_path)
        np.random.shuffle(self.ground_truth)
        n_train = int(np.floor(len(self.ground_truth) * 0.6))
        n_val = int(np.floor(len(self.ground_truth) * 0.2))

        self.train = self.ground_truth[:n_train]
        self.validate = self.ground_truth[n_train:n_train + n_val]
        self.test = self.ground_truth[n_train + n_val:]


class DataBase:

    def __init__(self, db_path):
        self.db_cursor = None
        self.columns = ["title", "artists", "isrcs", "contributors"]
        self.get_database(db_path)

    def get_database(self, db_path):
        if not os.path.exists(db_path):
            raise FileNotFoundError
        self.db_cursor = sqlite3.connect(db_path).cursor()

    def get_sr_by_srid(self, sr_id):
        self.db_cursor.execute("select title, artists, isrcs, contributors from soundrecording where sr_id = ?",
                               [sr_id])
        rows = self.db_cursor.fetchall()

        if not rows:
            raise KeyError

        if len(rows) > 1:
            raise KeyError

        match = {}
        for item, column_name in zip(rows[0], self.columns):
            match[column_name] = item

        return match

# data = Data("groundtruth.csv", "db.db")
# stuff = data.get_sr_by_srid("spotify_apidsr__2NbYAPqE6FTyQte9kW4vgr")
# cosa = 0
