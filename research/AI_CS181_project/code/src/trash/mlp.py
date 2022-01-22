from sklearn.neural_network import MLPRegressor
from trash.base import baseline
def _mlp(TF_ID):
    model = MLPRegressor(hidden_layer_sizes=(16),  activation='relu', solver='adam', max_iter=10)
    baseline(model,TF_ID)

if __name__ == '__main__':
    _mlp()
