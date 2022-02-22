# Product Review Classification
A simple project to classify a sample data-set of product reviews ([Digikala](https://www.digikala.com/opendata/)). Existing classes are: _Recommended_, _Not Recommended_, and _Neutral_.

### Instructions:
- First do a `$ pip install -r requirements.txt` to install the required modules.
- To run the project from system's terminal, you might need to activate virtual environment of this project:
<br>`$ source bin/bin/activate`
- Training: To train a model, run `$ python main.py train model`. For example, to train the BiLSTM model run `$ python main.py train bilstm`.
- Testing: To test a model with provided data-set, run `$ python main.py test model`. Similar to the train command, run `$ python main.py test bilstm` to test the trained BiLSTM model.
- Available models are: **RNN**, **LSTM**, and **BiLSTM**. Input to the command-line is automatically converted to lowercase.

### Project Structure:
- All training/test data are located in the data directory. 
- `Config.json` can be used to change parameters of models, default options, training iterations, etc.


