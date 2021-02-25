from __future__ import unicode_literals
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from utils import save_model, load_tokens
from hazm import *
import re


class TextProcessor:
    """
    Provide methods for pre-processing and text conversion.

    Attributes:
        tokens (list):
        sequences (list):
    """

    def __init__(self):
        self.tokens = []
        self.sequences = []

    def text_encoding(self, text, max_words, max_len, is_test=False):
        """
        Convert text documents to integer sequences. Documents will be having a padding of a max length of max_len words.

        :param str text: Input documents
        :param int max_words: The maximum number of words to keep. Only the most common num_words-1 words will be kept
        :param int max_len: Length of words for padding the documents
        :param is_test: If true, load extracted tokens in training phase. Otherwise create the tokens from input text
        :return:
        """

        if is_test:
            tokenizer = load_tokens("tokens.pickle")
        else:
            tokenizer = Tokenizer(num_words=max_words)
            tokenizer.fit_on_texts(text)
            self.tokens = tokenizer
            save_model("tokens.pickle", self.tokens)

        sequences = tokenizer.texts_to_sequences(text)
        sequences = pad_sequences(sequences, maxlen=max_len)
        self.sequences = sequences

    def preprocess_text(self, text):
        """
        Pre-process text by normalizing, removing and replacing some tokens.

        :param text: Input data text
        :return: Normalized and cleaned input data
        """

        text = text.tolist()
        normalizer = Normalizer()
        normalized_data = []
        # punctuation = r"""!"#$%&'()*+,،-./:;<=>?@[\]^_`{|}~"""  # Alternative method to deal with punctuations
        for item in text:
            if bool(re.search(r'\d', item)):
                # Replace digital characters with some special tokens
                item = re.sub('[0-9]{5,}', '#####', item)
                item = re.sub('[0-9]{4}', '####', item)
                item = re.sub('[0-9]{3}', '###', item)
                item = re.sub('[0-9]{2}', '##', item)
                item = re.sub('[0-9]', '#', item)
            item = re.sub('[A-Za-z]', '#!', item)  # Replace non-persian characters with a token
            item = re.sub(r'[^\w\s]', '', item)  # Remove punctuations
            # item = item.translate(str.maketrans('', '', punctuation))  # Alternative method to deal with punctuations

            normalized_item = normalizer.normalize(item)
            normalized_data.append(normalized_item)
        return normalized_data
