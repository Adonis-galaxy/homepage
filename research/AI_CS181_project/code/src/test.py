# Modified by xuyt1 on 2022/1/10
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import os
mapping=["anger","joy","optimism","sadness"]
def testing(model,test_data,test_labels,num_test,row_test_data,dnn_type = 0):
    label=[0,0,0,0]
    hit_label=[0,0,0,0]
    correct_prediction=[[],[],[],[]]
    wrong_prediction=[[],[],[],[]]
    # TP = FP = FN = TN = precision = recall = F1_score =[0,0,0,0]
    TP = [0,0,0,0]
    FP = [0,0,0,0]
    FN = [0,0,0,0]
    TN = [0,0,0,0]
    precision = [0,0,0,0]
    recall = [0,0,0,0]
    F1_score = [0,0,0,0]
    true_test=0
    if dnn_type == 0:
        predict_test = model.predict(test_data.T)
        if type(model) == MLPRegressor:
            predict_test = predict_test.argmax(1)
    else:
        predict_raw = model.predict(test_data.reshape(num_test,-1))
        predict_test = []
        for i in predict_raw:
            predict_test.append(np.argmax(i))

    for i in range(num_test):
        label[test_labels[i]] += 1
        if predict_test[i] == test_labels[i]:
            true_test += 1
            hit_label[test_labels[i]] += 1
            correct_prediction[predict_test[i]].append(row_test_data[i])
            TP[predict_test[i]] += 1
            # print("TP",predict_test[i],TP[predict_test[i]])
            for h in range(4):
                if h != predict_test[i]:
                    TN[h] +=1
        else:
            wrong_prediction[predict_test[i]].append((row_test_data[i],mapping[test_labels[i]]))
            FN[test_labels[i]]+=1
            FP[predict_test[i]]+=1
            for h in range(4):
                if h != predict_test[i] and h != test_labels[i]:
                    TN[h] +=1
    # print("TP",TP)
    for i in range(4):
        precision[i] = TP[i] / (TP[i] + FP[i])
        recall[i] = TP[i] / (TP[i] + FN[i])
        F1_score[i] = 2 * (precision[i] * recall[i])/(precision[i] + recall[i])

    print("test acc:", true_test/num_test)
    print("avg acc:", (hit_label[0] / label[0] + hit_label[1] / label[1] + hit_label[2] / label[2] + hit_label[3] / label[3]) / 4)
    print("anger prediction acc:",hit_label[0] / label[0])
    print("joy prediction acc:",hit_label[1] / label[1])
    print("optimism prediction acc:",hit_label[2] / label[2])
    print("sadness prediction acc:",hit_label[3] / label[3])

    # print("TP",TP)

    model_names=['knn','svm','mlp','naive_bayes','rnn','lstm','transformer']
    motions = ['anger','joy','optimism','sadness']
    model_index=None
    if dnn_type == 0:
        if type(model) == sklearn.neighbors._classification.KNeighborsClassifier:
            model_index = 0
        elif type(model) == sklearn.svm._classes.LinearSVC:
            model_index = 1
        elif type(model) == sklearn.neural_network._multilayer_perceptron.MLPRegressor:
            model_index = 2
        elif type(model) == sklearn.naive_bayes.ComplementNB:
            model_index = 3
    elif dnn_type == 1:
        model_index = 4
    elif dnn_type == 2:
        model_index = 5
    elif dnn_type == 3:
        model_index = 6
    else:
        raise ValueError("Invalid model type!")

    for k in range(len(motions)):
        with open(r"./example/"+model_names[model_index] + r"/"+motions[k]+r"/correct.txt","w",encoding="utf8") as f:
            try:
                for i in range(10):
                    f.write(" ".join(correct_prediction[k][i]))
                    f.write("\n")
            except IndexError:
                pass
        with open(r"./example/"+model_names[model_index] + r"/"+motions[k]+r"\wrong.txt","w",encoding="utf8") as f:
            try:
                for i in range(100):
                    f.write(" ".join(wrong_prediction[k][i][0]))
                    f.write(" || Ground truth: ")
                    f.write(wrong_prediction[k][i][1])
                    f.write("\n")
            except IndexError:
                pass
        with open(r"./example/"+model_names[model_index] + r"/"+motions[k]+r"/F1_score.txt","w",encoding="utf8") as f:
            f.write(str(F1_score[k]))
    # os.system(r"touch ./example/"+model_names[model_index] + r"/m_F1.txt")

    with open(r"./example/"+model_names[model_index] + r"/m_F1.txt","w",encoding="utf8") as f:
        f.write(str(sum(precision)/4))
    return true_test/num_test
