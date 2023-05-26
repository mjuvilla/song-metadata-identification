from data import DataBase, GroundTruth
from features import data_to_features
from measurements import compute_prf, compute_accuracy
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

    threshold = 64.0

    # train accuracy
    y_pred = []
    for item in x:
        y_pred.append(int(np.mean(item) > threshold))
    train_acc = compute_accuracy(y, y_pred)
    print(f"Train accuracy: {train_acc:.3f}\n")

    # validation tests
    x_val, y_val = data_to_features(ground_truth_handler.validate, database_handler)
    y_pred = []
    for item in x_val:
        y_pred.append(int(np.mean(item) > threshold))

    val_acc = compute_accuracy(y_val, y_pred)
    val_precision, val_recall, val_fscore = compute_prf(y_val, y_pred)

    print(f"Validation accuracy: {val_acc:.3f}, Validation precision: {val_precision:.3f}, "
          f"Validation recall: {val_recall:.3f}, Validation f-score: {val_fscore:.3f}")
