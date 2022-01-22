# from new_keras_rnn import _rnn_and_lstm
# from transformer import _transformer
TF_ID = False
from Naive_Bayes_Classifiers import Naive_Bayes_Classifiers
from DNN_Based_Classification import _RNN, _LSTM, _Transformer
Naive_Bayes_Classifiers._knn(TF_ID)
Naive_Bayes_Classifiers._svm(TF_ID)
Naive_Bayes_Classifiers._mlp(TF_ID)
Naive_Bayes_Classifiers._naive_nayes_mle(TF_ID)
_RNN()
_LSTM()
_Transformer()
