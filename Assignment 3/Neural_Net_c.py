
import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split 

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def Tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    return np.maximum(0.1*x, x)

def d_sigmoid(x):
    g = sigmoid(x)
    return np.multiply(g, 1-g)

def d_Tanh(x):
    g = Tanh(x)
    return (1-np.square(g))

def d_relu(x):
    g = relu(x)
    return (1.0*(g>0))

def d_leaky_relu(x):
    g = leaky_relu(x)
    d = np.zeros_like(g)
    d[g<=0] = 0.1
    d[g>0] = 1
    return d

def softmax(x):
    return np.exp(x)/np.exp(x).sum(axis=1).reshape(np.exp(x).shape[0],1)

def loss(pred, given):
    N = pred.shape[0]
    pred = np.clip(pred, 1e-12, 1 - 1e-12)
    ce = - np.sum(given * np.log(pred))
    ce = ce/N
    return ce

def initialize(n_layers, h_layers, feat):
    B = []
    W = []
    B.append(np.zeros(h_layers[0]))
    W.append(np.random.randn(feat, h_layers[0])*np.sqrt(2/(feat + h_layers[0])))
    for i in range(n_layers-1):
        B.append(np.zeros(h_layers[i+1]))
        W.append(np.random.randn(h_layers[i], h_layers[i+1])*np.sqrt(2/(h_layers[i] + h_layers[i+1])))
    B.append(np.zeros(10))
    W.append(np.random.randn(h_layers[-1], 10)*np.sqrt(2/(10 + h_layers[-1])))
    return W, B

def forward_propagation(X, W, B):
    Z = []
    A = []
    Z.append((np.dot(X, W[0]) + B[0]))
    A.append(sigmoid((np.dot(X, W[0]) + B[0])))
    for i in range(len(W)-2):
        Z.append((np.dot(A[i], W[i+1]) + B[i+1]))
        A.append(sigmoid((np.dot(A[i], W[i+1]) + B[i+1])))
    Z.append((np.dot(A[-1], W[-1]) + B[-1]))
    A.append(softmax((np.dot(A[-1], W[-1]) + B[-1])))
    return Z, A

def backward_propagation(X, Y, W, B, Z, A, batch_size, learning_rate):
    h = len(W)
    dz = Y - A[-1]
    for i in range(h - 1):
        dw = np.dot(A[-i-2].T, dz/batch_size)
        db = np.sum(dz/batch_size, axis=0)
        da = np.dot(dz, W[-i-1].T)
        dz = np.multiply(da, d_sigmoid(Z[-i-2]))
        W[-i-1] = W[-i-1] + learning_rate*dw
        B[-i-1] = B[-i-1] + learning_rate*db
    dw = np.dot(X.T, dz/batch_size)
    db = np.sum(dz/batch_size, axis=0)
    W[0] = W[0] + learning_rate*dw
    B[0] = B[0] + learning_rate*db
    return W, B

def createbatch(x, y, size):
    batches = []
    n_batch = x.shape[0]//size
    for i in range(n_batch):
        x_m = x[i*size:(i+1)*size,:] 
        y_m = y[i*size:(i+1)*size,:]
        batches.append((x_m,y_m))
    return batches

def validationNeural(X, Y, n_layers, h_layers, iters, b_size, learning_rate):
    accuracy_list = []
    accuracy_list2 = []
    error_list = []
    error_list2 = []
    W, B = initialize(n_layers, h_layers, 784)
    alpha0 = learning_rate
    train, test, train_out, test_out = train_test_split(X, Y, test_size = 0.2)
    p = 1
    while(p <= iters):
        alpha = alpha0/pow(p, 1/3)
        p += 1
        batches = createbatch(train, train_out, b_size)
        for batch in batches:
            x_mini, y_mini = batch
            Z, A = forward_propagation(x_mini, W, B)
            W, B = backward_propagation(x_mini, y_mini, W, B, Z, A, b_size, alpha)
    Z_final, A_final = forward_propagation(train, W, B)
    pred_train = np.argmax(A_final[-1], axis = 1)
    given_train = np.argmax(train_out, axis = 1)
    accuracy_list.append(metrics.accuracy_score(pred_train, given_train))
    error_list.append(loss(A_final[-1], train_out))
    Z_final2, A_final2 = forward_propagation(test, W, B)
    pred_test = np.argmax(A_final2[-1], axis = 1)
    given_test = np.argmax(test_out, axis = 1)
    accuracy_list2.append(metrics.accuracy_score(pred_test, given_test))
    error_list2.append(loss(A_final2[-1], test_out))
    return sum(accuracy_list)/len(accuracy_list), sum(accuracy_list2)/len(accuracy_list2), sum(error_list)/len(error_list), sum(error_list2)/len(error_list2)

if __name__ == "__main__":
    f = open("2017EE10500_PCA.csv")
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
        a = int(rows[i][25])
        Y[i][a] = 1
        
    n_layers = 1
    h_layers = [15]
    max_iter = 5000
    b_size = 100
    i = 1
    alpha0 = 1
    W, B = initialize(n_layers, h_layers, 25)
    train, test, train_out, test_out = train_test_split(X, Y, test_size = 0.2)
    while(i <= max_iter):
        i += 1
        alpha = alpha0/pow(i, 1/3)
        batches = createbatch(train, train_out, b_size)
        for batch in batches:
            x_mini, y_mini = batch
            Z, A = forward_propagation(x_mini, W, B)
            W, B = backward_propagation(x_mini, y_mini, W, B, Z, A, b_size, alpha)
    Z_final, A_final = forward_propagation(test, W, B)
    pred = np.argmax(A_final[-1], axis = 1)
    given = np.argmax(test_out, axis = 1)
    acc = metrics.accuracy_score(pred, given)
    
    
    """a = []
    b = []
    c = []
    d = []
    i = 15
    while(i <= 15):
        h_layers[0] = i
        print(h_layers)
        aa, ba, ce, de = validationNeural(X, Y, n_layers, h_layers, 5000, 100, 1)
        a.append(aa)
        b.append(ba)
        c.append(ce)
        d.append(de)
        i += 5



    print('a = [' + " ".join(str(x) for x in a) + ']')
    print('b = [' + " ".join(str(x) for x in b) + ']')
    print('c = [' + " ".join(str(x) for x in c) + ']')
    print('d = [' + " ".join(str(x) for x in d) + ']')"""
    
    
#Gradient descent    
    """max_iter = 5000
    b_size = 100
    i = 1
    alpha0 = 1
    W, B = initialize(n_layers, h_layers, 784)
    error_list = []
    while(i <= max_iter):
        i += 1
        alpha = alpha0/pow(i, 1/3)
        batches = createbatch(X, Y, b_size)
        for batch in batches:
            x_mini, y_mini = batch
            Z, A = forward_propagation(x_mini, W, B)
            W, B = backward_propagation(x_mini, y_mini, W, B, Z, A, b_size, alpha)
        Z_final, A_final = forward_propagation(X, W, B)
        pred = np.argmax(A_final[-1], axis = 1)
        given = np.argmax(Y, axis = 1)
        print(loss(A_final[-1], Y))
        error_list.append(loss(A_final[-1], Y))"""