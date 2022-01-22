import pandas as pd
import re
import numpy as np
from wordsegment import load, segment


def data_loader(text):
    train_text = text
    train_data = []
    l = 0
    for sentence in train_text:
        words = re.split(r'[“\[\]\-:;,.@!#?*–~()\s]\s*', sentence)
        train_data.append([])
        for word in words:
            if word == '':
                continue
            start = 0
            for i in range(len(word)):
                if i == 0 and word[i] == '\'':
                    start = start + 1
                    continue
                if i == len(word) - 1 and word[i] == '\'':
                    word = word[:-1]
                    break
                if word[i:i+2] == '\\n':
                    if 0 < i - start:
                        train_data[l].append(word[start:i].lower())
                    start = i + 2
                    i = i + 1
                    continue
                if word[i] == '/':
                    if 0 < i - start:
                        train_data[l].append(word[start:i].lower())
                    start = i + 1
                    continue
                if ord(word[i]) > 10000:
                    if word[i] != '❤':
                        if 0 < i - start:
                            train_data[l].append(word[start:i].lower())
                        train_data[l].append(word[i])
                        start = i + 1
                    else:
                        if 0 < i - start:
                            train_data[l].append(word[start:i].lower())
                        train_data[l].append(word[i:i+2])
                        start = i + 2
                        i = i + 1
                        continue
                if '0' <= word[i] <= '9':
                    if 0 < i - start:
                        train_data[l].append(word[start:i].lower())
                    while '0' <= word[i] <= '9':
                        i = i + 1
                        if i == len(word):
                            break
                        continue
                    i = i - 1
                    start = i + 1
            i = len(word)
            if start != i:
                if 0 < i - start:
                    train_data[l].append(word[start:i].lower())
        l += 1
    load()
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            p = segment(train_data[i][j])
            if len(p) > 1:
               train_data[i][j] = p[0]
            train_data[i]+= p[1:]
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            if train_data[i][j][-1:] == 's':
                train_data[i][j] = train_data[i][j][:-1]
            if train_data[i][j][-1:] == 'e':
                train_data[i][j] = train_data[i][j][:-1]
            try:
                if train_data[i][j][-2:] == 'ly':
                    train_data[i][j] = train_data[i][j][:-2]
                    continue
                if train_data[i][j][-2:] == 'ed':
                    train_data[i][j] = train_data[i][j][:-2]
                    continue
                if train_data[i][j][-2:] == 'al':
                    train_data[i][j] = train_data[i][j][:-2]
                    continue
                if train_data[i][j][-2:] == 'er':
                    train_data[i][j] = train_data[i][j][:-2]
                    continue
                if train_data[i][j][-3:] == 'ing':
                    train_data[i][j] = train_data[i][j][:-3]
                    continue
                if train_data[i][j][-3:] == 'ion':
                    train_data[i][j] = train_data[i][j][:-3]
                    continue
                if train_data[i][j][-3:] == 'ful':
                    train_data[i][j] = train_data[i][j][:-3]
                    continue
                if train_data[i][j][-3:] == 'nes':
                    train_data[i][j] = train_data[i][j][:-3]
                    continue
                if train_data[i][j][-4:] == 'ment':
                    train_data[i][j] = train_data[i][j][:-4]
                    continue
            except:
                pass
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            k = 0
            while k < len(train_data[i][j]) - 1:
                while train_data[i][j][k] == train_data[i][j][k + 1]:
                    if k + 1 == len(train_data[i][j]):
                        train_data[i][j] = train_data[i][j][:k + 1]
                    else:
                        train_data[i][j] = train_data[i][j][:k + 1] + train_data[i][j][k + 2:]
                    if k >= len(train_data[i][j]) - 1:
                        break
                k += 1

    return train_data


def histogram_building(text, bag=1):
    histogram = {}
    for i in range(bag):
        for words in text:
            for j in range(len(words) - i):
                temp = ''
                for k in range(i + 1):
                    temp += words[j]
                if temp not in histogram.keys():
                    histogram[temp]=1
                else:
                    histogram[temp] += 1
    histogram = pd.Series(histogram)
    return histogram


def load_text(file_name):
    lst=[]
    with open("../data/"+file_name+".txt", encoding='utf8') as f:
        for line in f:
            lst.append(line.strip('\n'))
    data = data_loader(lst)
    return data


def load_label(file_name):
    lst=[]
    with open("../data/"+file_name+".txt", errors='ignore') as f:
        for line in f:
            lst.append(int(line.strip('\n')))
    return lst



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


def feature_extractor():
    data = load_text("train_text")
    histogram = histogram_building(data)
    feature = list(histogram.index)
    l = len(feature)
    num = [0 for _ in range(l)]
    for i in range(l):
        for sentence in data:
            if feature[i] in sentence:
                num[i] += 1
    scores = []
    train_data, num_train = text_preprocessing(data, feature, l, bag=1)
    for i in range(len(data)):
        scores.append([])
        s = []
        for j in range(len(data[i])):
            if data[i][j] not in s:
                s.append(data[i][j])
            else:
                continue
            word = data[i][j]
            index = feature.index(word)
            idf = np.log(l/(num[index] + 1))
            TF = train_data[index, i]
            TF_IDF = TF * idf
            scores[i].append((TF_IDF, data[i][j]))
        scores[i].sort(key=lambda x: -x[0])
    vec = np.zeros(shape=(l, len(data)))
    for i in range(len(data)):
        s = []
        for j in range(min(10, len(scores[i]))):
            index = scores[i][j][1]
            s.append(index)
        words = data[i]
        for j in range(len(words)):
            try:
                if words[j] in s:
                    index = feature.index(words[j])
                    vec[index, i] += 1
            except ValueError:  # some data in val may not appear in training set
                pass
    return vec, len(data)


def text_preprocessing(text,tokens,num_feature, bag=1):
    num_data = len(text)
    data = np.zeros(shape=(num_feature, num_data))
    # print(train_data.shape) # (?, 3257)
    for l in range(bag):
        for i in range(num_data):
            words = text[i]
            for j in range(len(words) - l):
                temp = ''
                for k in range(l + 1):
                    temp += words[j]
                try:
                    if temp in tokens:
                        index = tokens.index(temp)
                        data[index, i] += 1
                except ValueError: # some data in val may not appear in training set
                    pass
    return data, num_data
