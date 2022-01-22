import numpy as np
def label_preprocessing(num_train,train_labels):
    train_labels_processed = np.zeros(shape=(num_train,4))
    for i in range(num_train):
        train_labels_processed[i,int(train_labels[i])]=1
    return train_labels_processed
"""
before preprocess:[0,2,3,1]
after preprocess:
[
    [1,0,0,0]
    [0,0,1,0]
    [0,0,0,1]
    [0,1,0,0]
]
"""