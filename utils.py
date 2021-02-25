import numpy as np
import pandas as pd
from constants import *
import tensorflow as tf
import pickle
import json
import argparse
from sklearn.model_selection import train_test_split
import sys


def read_data(path, show_labels=False):
    """
    Read the data file and converts to a list.

    :param str path: Path to the document
    :param bool show_labels: Show all labels used in data
    :return: DataFrame
    """

    data = pd.read_csv(path)
    data = data.dropna(subset=[VALUE])  # Remove null values
    data = data.loc[:, [VALUE, LABEL]]  # Keep these columns
    data = data.loc[(data[LABEL] != '\\N')]  # Keeping only labeled comments

    if show_labels:
        print(data[LABEL].unique())
    return data


def normalize_data_distribution(data):
    """
    Equally distribute data according to the minimum available class (label).

    :param pandas.core.frame.DataFrame data: Input data to distributed
    :return: Equally distributed data
    """

    dist = data.groupby(LABEL).nunique()
    min_label = min(dist[VALUE])  # Get the value of the rarest available label
    normalized_data = data.groupby(LABEL).head(n=min_label)  # Now data are equally distributed
    return normalized_data


def encode_labels(labels):
    """
    Encode labels to categorical type with Keras.

    :param pandas.core.series.Series labels: Original labels of input data
    :return: Encoded labels
    """

    labels_array = np.array(labels)
    y = []
    for i in range(len(labels_array)):
        if labels_array[i] == 'no_idea':
            y.append(0)
        if labels_array[i] == 'not_recommended':
            y.append(1)
        if labels_array[i] == 'recommended':
            y.append(2)
    y = np.array(y)
    labels_array = tf.keras.utils.to_categorical(y, 3, dtype="float32")
    del y
    return labels_array


def save_model(fn, obj):
    """
    Save a given model for later uses.

    :param str fn: Name of file to be saved
    :param obj: Data to be saved
    :return:
    """

    with open(fn, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def read_config(fn, perm="r"):
    """
    Read config file which consists parameters used in program.

    :param str fn: Name of config file (config.json)
    :param str perm: Mode in which the file is opened
    :return: Config file paramteres
    """

    with open(fn, perm) as file:
        config = json.load(file)
    return config


def load_tokens(fn, perm="rb"):
    """
    Read config file which consists parameters used in program.

    :param str fn: Name of file
    :param str perm: Mode in which the file is opened
    :return: Keras tokenizer object consisting extracted tokens
    """

    with open(fn, perm) as file:
        tokens = pickle.load(file)
    return tokens


def split_data(data, labels):
    """
    Split dataset to train and test sets.

    :param data: Input data
    :param labels: Corresponding labels of data
    :return: List containing train-test split of input data
    """

    X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=0)
    return X_train, X_test, y_train, y_test


def init_app(args):
    """
    The entry of program. Check if program is launched from terminal & Process data.

    :param args: Arguments passed from CLI
    :return: Processed data and mode of program (Train or test). Exit if none.
    """

    print("Initializing...")
    is_cli_mode = sys.stdout.isatty()
    cli_args = cli_mode()
    mode = (cli_args.mode.lower() if is_cli_mode else args["mode"])
    model = (cli_args.model.lower() if is_cli_mode else args["model"])
    if mode == "train":
        data = read_data(TRAIN_DATA)
        data = normalize_data_distribution(data)
        return data, mode, model
    elif mode == "test":
        data = read_data(TEST_DATA)
        return data, mode, model
    # elif mode == "stats":
    #     show_model_info(args.model_name)
    #     exit()
    else:
        sys.exit("Selected mode is wrong. (train or test)")
    # print("Loading {0} mode...".format(mode))


def cli_mode():
    """
    Configurations fo CLI to run program with. Will be set to default values if run in other environments.

    :return: Parsed arguments from CLI
    """

    parser = argparse.ArgumentParser(description="Manual to use this script:", usage="python main.py mode model")
    parser.add_argument('mode', type=str, nargs='?', default="test", help='Choose whether you want to train a model '
                                                                          'or test one')
    parser.add_argument('model', type=str, nargs='?', default="lstm", help='Choose the model you wish to train/test')
    args = parser.parse_args()
    return args

