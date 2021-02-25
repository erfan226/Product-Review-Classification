from utils import *
from text_processor import TextProcessor
from models import train_model, predict
from constants import *

# Config file (Parameters)
params = read_config("config.json")

# Read and process data
data, mode, model = init_app(params)

labels = encode_labels(data[LABEL])

# Pre-process input data and convert them
text = data[VALUE]
tp = TextProcessor()
text = tp.preprocess_text(text)

# Training/Testing
if mode == "train":
    text_sequences = tp.text_encoding(text, **params["text_process"])
    X_train, X_test, y_train, y_test = split_data(tp.sequences, labels)
    train_model(X_train, X_test, y_train, y_test, params["model"]["conf"]["show_plot"], model)
else:
    text_sequences = tp.text_encoding(text, is_test=True, **params["text_process"])
    predict(tp.sequences, labels, model, params["test"]["to_categorical_labels"])
