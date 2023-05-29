import os
from data import DataBase, GroundTruth
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from features import data_to_features

# matplotlib.use("TkAgg")


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    ground_truth_handler = GroundTruth("data/groundtruth.csv", val_split=0)
    database_handler = DataBase("data/db.db")

    x_train, y_train = data_to_features(ground_truth_handler.train, database_handler)
    x_test, y_test = data_to_features(ground_truth_handler.test, database_handler)

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(x_train, y_train)

    yp_train = clf.predict(x_train)
    yp_test = clf.predict(x_test)

    train_accuracy = accuracy_score(y_train, yp_train)
    test_accuracy = accuracy_score(y_test, yp_test)

    train_precision = precision_score(y_train, yp_train)
    test_precision = precision_score(y_test, yp_test)

    train_recall = recall_score(y_train, yp_train)
    test_recall = recall_score(y_test, yp_test)

    print(f"TRAIN - accuracy: {train_accuracy:.3}"
          f"\tprecision: {train_precision:.3}"
          f"\trecall: {train_recall:.3}")
    print(f"TEST - accuracy: {test_accuracy:.3}"
          f"\tprecision: {test_precision:.3}"
          f"\trecall: {test_recall:.3}")

    #
    # torch.save(model, 'soundrecording-classifier.pt')
    #
    # # plot results
    # # loss plot
    # f.add_subplot(2, 3, 1)
    # plt.plot(loss)
    # plt.title("Loss")
    #
    # # train acc plot
    # f.add_subplot(2, 3, 2)
    # plt.plot(train_acc)
    # plt.ylim([0, 1])
    # plt.title("Train accuracy")
    #
    # # validation acc plot
    # f.add_subplot(2, 3, 3)
    # plt.plot(val_acc)
    # plt.ylim([0, 1])
    # plt.title("Validation accuracy")
    #
    # # validation precision plot
    # f.add_subplot(2, 3, 4)
    # plt.plot(val_precision)
    # plt.ylim([0, 1])
    # plt.title("Validation precision")
    #
    # # validation recall plot
    # f.add_subplot(2, 3, 5)
    # plt.plot(val_precision)
    # plt.ylim([0, 1])
    # plt.title("Validation recall")
    #
    # # validation fscore plot
    # f.add_subplot(2, 3, 6)
    # plt.plot(val_precision)
    # plt.ylim([0, 1])
    # plt.title("Validation fscore")
    #
    # plt.show()
