from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn import svm
import sklearn
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from data_process import label_preprocessing
def training(model,train_data,train_labels,num_train,val_data,val_labels,num_val):
    print("\ntype of the model:",type(model))
    if type(model) == MLPRegressor:
        train_labels_processed = label_preprocessing(num_train,train_labels)
        model.fit(train_data.T, train_labels_processed)
        train_accuracy = sum(model.predict(train_data.T).argmax(1) == train_labels) / num_train
        print("train acc",train_accuracy)
    else:
        model.fit(train_data.T, train_labels)
        train_accuracy = sum(model.predict(train_data.T) == train_labels) / num_train
        print("\ntrain acc",train_accuracy)