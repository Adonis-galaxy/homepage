# Modified by xuyt1 on 2022/1/11
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn import svm
import sklearn
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import ComplementNB  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
import numpy as np
from sklearn.neural_network import MLPRegressor
import sys
from data_process import text_preprocessing
from data_process import label_preprocessing
from train import training
from test import testing
from data_process import load_text,load_label, feature_extractor
from data_process import histogram_building
from sklearn.svm import LinearSVC

class Naive_Bayes_Classifiers():
    def _knn(TF_ID):
        model = KNeighborsClassifier(n_neighbors=4)
        Naive_Bayes_Classifiers.baseline(model,TF_ID)
    def _svm(TF_ID):
        model = svm.LinearSVC(penalty='l1', dual=False, max_iter=5, tol=1e-3,random_state=38, fit_intercept=True)
        Naive_Bayes_Classifiers.baseline(model,TF_ID)
    def _mlp(TF_ID):
        model = MLPRegressor(hidden_layer_sizes=(16),  activation='relu', solver='adam', max_iter=10)
        Naive_Bayes_Classifiers.baseline(model,TF_ID)
    def _naive_nayes_mle(TF_ID):
        model = ComplementNB(alpha=1.1)
        Naive_Bayes_Classifiers.baseline(model,TF_ID)
    def baseline(model = LinearSVC(), TF_ID=False):
        train_text = load_text("train_text")
        train_labels = np.array(load_label("train_labels"))
        val_text = load_text("val_text")
        val_labels = np.array(load_label("val_labels"))
        test_text = load_text("test_text")
        test_labels = np.array(load_label("test_labels"))
        histogram = histogram_building(train_text, bag=1)
        num_feature = len(histogram) # 12887
        if TF_ID:
            pass
            #tokens = feature_set()
        else:
            tokens = list(histogram.index)
        print("dimension = ", len(tokens))
        train_data,num_train = feature_extractor()
        val_data,num_val = text_preprocessing(val_text,tokens,num_feature, bag=1)
        test_data,num_test = text_preprocessing(test_text,tokens,num_feature, bag=1)
        training(model,train_data,train_labels,num_train,val_data,val_labels,num_val)
        test_acc=testing(model,test_data,test_labels,num_test,test_text)
        return test_acc
