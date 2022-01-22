import numpy as np
def text_preprocessing(text,tokens,num_feature, bag=1):
    num_data = len(text)
    data = np.zeros(shape=(num_feature, num_data))
    # print(train_data.shape) # (?, 3257)
    for l in range(bag):
        for i in range(num_data):
            words = text[i].split()
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