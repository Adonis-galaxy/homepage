import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from data_process import load_text,load_label
from data_process import histogram_building
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

# This load_text does not use the trick
def naive_load_text(file_name):
    lst=[]
    with open("../data/"+file_name+".txt", encoding='utf8') as f:
        for line in f:
            lst.append(line.strip('\n'))
    return lst

def _rnn_and_lstm(mode):
    train_text = load_text("train_text")
    train_labels = np.array(load_label("train_labels"))
    val_text = load_text("val_text")
    val_labels = np.array(load_label("val_labels"))
    test_text = load_text("test_text")
    test_labels = np.array(load_label("test_labels"))
    histogram = histogram_building(train_text)
    num_feature = 12887  # 12887
    tokens = list(histogram.index)


    '''Map the text into word vectors and 
    then pad the sequences to the same length 
    for further processing'''
    max_words = 12887
    max_len = 150
    tok = Tokenizer(num_words=max_words)
    Y_train = to_categorical(train_labels)
    Y_val = to_categorical(val_labels)
    Y_test = to_categorical(test_labels)

    tok.fit_on_texts(train_text)
    train_seq = tok.texts_to_sequences(train_text)
    val_seq = tok.texts_to_sequences(val_text)
    test_seq = tok.texts_to_sequences(test_text)
    train_seq_mat = sequence.pad_sequences(train_seq, maxlen=max_len)
    val_seq_mat = sequence.pad_sequences(val_seq, maxlen=max_len)
    test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)


    '''SimpleRNN and LSTM are both implemented in this file.
    Change the name of that line leads to the switch.'''
    inputs = keras.layers.Input(name='inputs', shape=[max_len])
    layer = keras.layers.Embedding(max_words+1, 128, input_length=max_len)(inputs)
    if mode == 0:  # SimpleRNN Mode
        layer = keras.layers.SimpleRNN(128)(layer)  # For switching between SimpleRNN and LSTM
    elif mode == 1:   # LSTM Mode
        layer = keras.layers.LSTM(128)(layer)
    else:
        raise ValueError("Invalid mode!")
    layer = keras.layers.Dense(128, activation="tanh")(layer)
    layer = keras.layers.Dropout(0.2)(layer)  # For RNN, the dropout rate should be higher
    layer = keras.layers.Dense(4, activation="softmax")(layer)
    model = keras.Model(inputs=inputs, outputs=layer)

    # model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

    '''The convergence of simple RNN is too slow that
    early stopping normally exits incorrectly.'''
    # For simple RNN, batch_size = 64
    #For RSTM, batch_size = 128
    model_fit = model.fit(train_seq_mat, Y_train, batch_size=64, epochs=10,
                          validation_data=(val_seq_mat, Y_val),
                         )
    return model

if __name__ == "__main__":
    mode = 0
    _rnn_and_lstm(mode)
