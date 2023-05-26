from data import DataBase, GroundTruth
from features import data_to_features
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

if __name__ == "__main__":
    ground_truth_handler = GroundTruth("data/groundtruth.csv")
    database_handler = DataBase("data/db.db")

    x, y = data_to_features(ground_truth_handler.train, database_handler)

    # separate between valid and invalid samples
    valid_samples = []
    invalid_samples = []
    for item, tag in zip(x, y):
        if tag:
            valid_samples.append(item)
        else:
            invalid_samples.append(item)

    valid_samples = np.array(valid_samples)
    invalid_samples = np.array(invalid_samples)

    # compute mean and std
    mean_valid_score = np.mean(np.mean(valid_samples, axis=1))
    std_valid_score = np.std(np.mean(valid_samples, axis=1))
    mean_invalid_score = np.mean(np.mean(invalid_samples, axis=1))
    std_invalid_score = np.std(np.mean(invalid_samples, axis=1))

    print(f"Valid samples -> mean score: {mean_valid_score:.3f}, std score: {std_valid_score:.3f}")
    print(f"Invalid samples -> mean score: {mean_invalid_score:.3f}, std score: {std_invalid_score:.3f}")

    # These are the results for the previous step, so we set the threshold to 64.0
    # Valid samples -> mean score: 81.320, std score: 15.178
    # Invalid samples -> mean score: 48.144, std score: 14.687
    threshold = 64.0

    # train accuracy
    y_pred = []
    for item in x:
        y_pred.append(int(np.mean(item) > threshold))
    train_acc = accuracy_score(y, y_pred)
    print(f"Train accuracy: {train_acc:.3f}")

    # validation tests
    x_val, y_val = data_to_features(ground_truth_handler.validate, database_handler)
    y_pred = []
    for item in x_val:
        y_pred.append(int(np.mean(item) > threshold))

    val_acc = accuracy_score(y_val, y_pred)
    val_precision = precision_score(y_val, y_pred)
    val_recall = recall_score(y_val, y_pred)
    val_fscore = f1_score(y_val, y_pred)

    print(f"Validation accuracy: {val_acc:.3f}, Validation precision: {val_precision:.3f}, "
          f"Validation recall: {val_recall:.3f}, Validation f-score: {val_fscore:.3f}")
