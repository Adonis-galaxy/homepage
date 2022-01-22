from new_keras_rnn import  _rnn_and_lstm
from transformer import _transformer
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from data_process import load_text,load_label
from data_process import histogram_building
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from test import testing
def _LSTM():
    train_text = load_text("train_text")
    test_text = load_text("test_text")
    test_labels = np.array(load_label("test_labels"))

    '''Map the text into word vectors and 
    then pad the sequences to the same length 
    for further processing'''

    max_words = 12887
    max_len = 150
    tok = Tokenizer(num_words=max_words)
    Y_test = to_categorical(test_labels)
    tok.fit_on_texts(train_text)
    test_seq = tok.texts_to_sequences(test_text)
    test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)

    model = _rnn_and_lstm(1)
    testing(model, test_seq_mat, test_labels, len(test_labels), test_text, 2)

def _RNN():
    train_text = load_text("train_text")
    test_text = load_text("test_text")
    test_labels = np.array(load_label("test_labels"))

    '''Map the text into word vectors and 
    then pad the sequences to the same length 
    for further processing'''

    max_words = 12887
    max_len = 150
    tok = Tokenizer(num_words=max_words)
    Y_test = to_categorical(test_labels)
    tok.fit_on_texts(train_text)
    test_seq = tok.texts_to_sequences(test_text)
    test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)

    model = _rnn_and_lstm(0)
    testing(model, test_seq_mat, test_labels, len(test_labels), test_text, 1)

def _Transformer():
    train_text = load_text("train_text")
    test_text = load_text("test_text")
    test_labels = np.array(load_label("test_labels"))

    '''Map the text into word vectors and 
    then pad the sequences to the same length 
    for further processing'''

    max_words = 12887
    max_len = 150
    tok = Tokenizer(num_words=max_words)
    Y_test = to_categorical(test_labels)
    tok.fit_on_texts(train_text)
    test_seq = tok.texts_to_sequences(test_text)
    test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)

    model = _transformer()
    testing(model, test_seq_mat, test_labels, len(test_labels), test_text, 3)

if __name__ == "__main__":
    _LSTM()
    # _Transformer()