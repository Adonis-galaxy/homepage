from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn import svm
import sklearn
from sklearn.svm import SVC
from sklearn import datasets       #导入数据模块
from sklearn.model_selection import train_test_split     #导入切分训练集、测试集模块
from sklearn.neighbors import KNeighborsClassifier
from trash.base import baseline
def _knn(TF_ID):
    model = KNeighborsClassifier(n_neighbors=4)
    baseline(model,TF_ID)
if __name__ == '__main__':
    _knn()
