from utils import read_config
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import numpy as np
from keras.models import Sequential, load_model
from keras import layers
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from keras import backend as K
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sb

# Config file (Parameters)
params = read_config("config.json")
# Limit test data predictions to a specific number
np.set_printoptions(edgeitems=params["test"]["limit_results"])
# Suppress warnings
if params["hide_info_log"]:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_model(X_train, X_test, y_train, y_test, show_plot, model_type="lstm"):
    """
    Train a given model.

    :param X_train: A set of data for training
    :param X_test:  A set of data for training
    :param y_train:  A set of data for training
    :param y_test:  A set of data for training
    :param show_plot: If true, show plot of models performance during training/validation
    :param model_type: Model to train (SimpleRNN, SingleLSTM, BiLSTM)
    :return:
    """

    model = Sequential()
    model_type = model_type.lower()
    opt = Adam(0.0015)

    model.add(layers.Embedding(params["model"]["embedding_layer"]["input_dimension"], params["model"]["embedding_layer"]["output_dimension"], input_length=params["model"]["embedding_layer"]["input_len"]))
    if model_type == "rnn":
        model.add(layers.SimpleRNN(**params["model"]["hidden_layer"], dropout=0.5))
    elif model_type == "lstm":
        model.add(layers.LSTM(**params["model"]["hidden_layer"], return_sequences=True, dropout=0.5))
        model.add(layers.LSTM(**params["model"]["hidden_layer"], dropout=0.5))
    elif model_type == "bilstm":
        model.add(layers.Bidirectional(layers.LSTM(**params["model"]["hidden_layer"], dropout=0.6)))
    else:
        sys.exit("Model not found! Options: RNN, LSTM, and BiLSTM")
    print("Training {0} model...".format(model_type))

    model.add(layers.Dense(**params["model"]["output_layer"]))
    model.compile(optimizer=opt, loss=params["model"]["conf"]["loss"], metrics=params["model"]["conf"]["metrics"])
    checkpoint = ModelCheckpoint("models/{0}.hdf5".format(model_type), monitor='val_accuracy', verbose=1, save_best_only=False,
                                  mode='auto', save_freq='epoch', save_weights_only=False)
    es = EarlyStopping(patience=40, verbose=1)
    history = model.fit(X_train, y_train, epochs=params["model"]["conf"]["epochs"], validation_data=(X_test, y_test), callbacks=[checkpoint, es])
    np.save('{0}_history.npy'.format(model_type), history.history)

    model.summary()
    plot_model(model, to_file='{0}_plot.png'.format(model_type), show_shapes=True, show_layer_names=True)

    if show_plot:
        plot_training(history)


def predict(data, labels, model="lstm", to_categorical=False):
    """
    Predict a set of test data given one of the trained models.

    :param data: Test data to predict
    :param labels: True labels of each data
    :param to_categorical: Converts to human-readable labels if true
    :return:
    """

    print("Loading model: models/{0}.{1}".format(model, 'hdf5'))
    models = ["rnn", "lstm", "bilstm"]
    if model not in models:
        sys.exit("Model not found! Options: RNN, LSTM, and BiLSTM")
    model = load_model("models/{0}.{1}".format(model, 'hdf5'))
    prediction = np.argmax(model.predict(data), axis=-1)
    if to_categorical:
        prediction = prediction.tolist()
        for i, label in enumerate(prediction):
            if label == 0:
                prediction[i] = "neutral"
            elif label == 1:
                prediction[i] = "not recommended"
            elif label == 2:
                prediction[i] = "recommended"
    print('Predicted classes:', prediction)
    eval_model(labels, prediction)


def plot_training(history):
    """
    Plot model's performance (Accuracy and loss) in training/validation phases.

    :param history: Performance values of accuracy and loss in each epoch
    :return:
    """

    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def eval_model(y_test, predictions):
    """
    Evaluate model on test and show its confusion matrix.

    :param y_test: True labels
    :param predictions: Predicted labels
    :return:
    """

    matrix = confusion_matrix(y_test.argmax(axis=1), predictions)
    conf_matrix = pd.DataFrame(matrix, index=['No idea', 'Not recommended', 'Recommended'],
                               columns=['No idea', 'Not recommended', 'Recommended'])
    # Normalizing
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(15, 15))
    sb.heatmap(conf_matrix, annot=True, annot_kws={"size": 15})
    plt.show()
