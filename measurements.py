import numpy as np


def compute_accuracy(y, y_pred):
    accuracy = sum(np.array(y_pred) == np.array(y)) / len(y)
    return accuracy.item()


# compute precision recall f-score
def compute_prf(y, y_pred):
    tp = 0
    fp = 0
    fn = 0

    for target, prediction in zip(y, y_pred):
        if target:
            if prediction:
                tp += 1
            else:
                fn += 1
        else:
            if prediction:
                fp += 1
            else:
                # tn, not necessary to compute
                pass
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        precision = 0
        recall = 0
        fscore = 0

    return precision, recall, fscore
