from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  
from sklearn.naive_bayes import BernoulliNB  
from sklearn.naive_bayes import ComplementNB  
from trash.base import baseline
def _naive_bayes(TF_ID):
    model = ComplementNB(alpha=1.1)
    baseline(model,TF_ID)

if __name__ == '__main__':
    _naive_bayes()

