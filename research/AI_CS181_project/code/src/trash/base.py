# Created by xuyt1 on 2021/12/21
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
import numpy as np
from sklearn.neural_network import MLPRegressor
import sys

from data_process.text_preprocess import text_preprocessing
from data_process.label_preprocess import label_preprocessing
from train import training
from test import testing
from data_process.fileloader import load_text,load_label
from data_process.build_histogram import histogram_building
from feature_generator import feature_set
from sklearn.svm import LinearSVC

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
        tokens = feature_set()
    else:
        tokens = list(histogram.index)
    print("dimension = ", len(tokens))
    train_data,num_train = text_preprocessing(train_text,tokens,num_feature, bag=1)
    val_data,num_val = text_preprocessing(val_text,tokens,num_feature, bag=1)
    test_data,num_test = text_preprocessing(test_text,tokens,num_feature, bag=1)
    training(model,train_data,train_labels,num_train,val_data,val_labels,num_val)
    test_acc=testing(model,test_data,test_labels,num_test,test_text)
    return test_acc