import os

from data import DataBase, GroundTruth
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from features import data_to_features

tf.random.set_seed(0)


# matplotlib.use("TkAgg")

def model_builder(hp):
    model = tf.keras.Sequential()
    for i in range(hp.Int('layers', 2, 6)):
        model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i), 10, 100, step=10),
                                        activation=hp.Choice('act_' + str(i), ['relu', 'sigmoid'])))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy(),
                           keras.metrics.Precision(),
                           keras.metrics.Recall()])
    return model


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    ground_truth_handler = GroundTruth("data/groundtruth.csv")
    database_handler = DataBase("data/db.db")

    x_train, y_train = data_to_features(ground_truth_handler.train, database_handler)
    x_val, y_val = data_to_features(ground_truth_handler.validate, database_handler)
    x_test, y_test = data_to_features(ground_truth_handler.test, database_handler)

    # prepare optimizer
    tuner = kt.Hyperband(model_builder,
                         objective="val_loss",
                         max_epochs=50,
                         factor=3,
                         directory='tuning_output',
                         project_name='song-metadata-id')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(x_train, y_train, epochs=50, validation_data=(x_val, y_val), callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(x_train, y_train, epochs=50)

    test_results = model.evaluate(x_test, y_test)

    print("Best model:")
    model.summary()
    print(f"TRAIN - accuracy: {history.history['binary_accuracy'][-1]:.3}"
          f"\tprecision: {history.history['precision'][-1]:.3}"
          f"\trecall: {history.history['recall'][-1]:.3}")
    print(f"TEST - accuracy: {test_results[1]:.3}"
          f"\tprecision: {test_results[2]:.3}"
          f"\trecall: {test_results[3]:.3}")

    model.save(os.path.join("output", "song-metadata-identification"))

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
