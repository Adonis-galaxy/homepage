import pandas as pd
def histogram_building(text, bag=1):
    histogram = {}
    for i in range(bag):
        for sentence in text:
            words = sentence.split() # tokenlize
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
