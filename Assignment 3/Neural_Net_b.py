
import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def loss(pred, given):
    N = pred.shape[0]
    pred = np.clip(pred, 1e-12, 1 - 1e-12)
    ce = - np.sum(given * np.log(pred))
    ce = ce/N
    return ce

if __name__ == "__main__":
    f = open("2017EE10500.csv")
    csv_f = csv.reader(f)
    rows = list(csv_f)
    n = len(rows)
    m = (len(rows[0])) - 1
    X = np.zeros([n, m])
    Y = np.zeros([n, 10])
    
    for i in range(n):
        for j in range(m):
            X[i][j] = float(rows[i][j])
    for j in range(m):
        t = X[:, j]
        maxi = np.max(t)
        mini = np.min(t)
        length = maxi - mini
        if length != 0:
            X[:, j] = ((X[:, j]) - mini)/length
            
    for i in range(n):
        a = int(rows[i][784])
        Y[i][a] = 1
        
    train, test, train_out, test_out = train_test_split(X, Y, test_size = 0.2)
    mlp = MLPClassifier(hidden_layer_sizes=(15), activation = 'logistic', solver = 'sgd', alpha = 0.001, batch_size = 100, learning_rate = 'invscaling', learning_rate_init = 1, power_t = 0.3333, max_iter = 5000, shuffle = False, verbose = False, momentum = 0)
    mlp.fit(train, train_out)
    pred = mlp.predict(test)
    pred2 = np.argmax(pred, axis = 1)
    test_out2 = np.argmax(test_out, axis = 1)
    acc = metrics.accuracy_score(pred2, test_out2)