
import numpy as np
import csv
import cvxopt
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import gridspec

def linearkernel(X):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i, x in enumerate(X):
        for j, y in enumerate(X):
            K[i, j] = np.dot(x.T, y)
    return K

def rbfkernel(X, gamma):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i, x in enumerate(X):
        for j, y in enumerate(X):
            K[i, j] = np.exp(-gamma * np.linalg.norm(x-y)**2)
    return K

def kfoldCrossValidationSVC(X, Y, k, c, g):
    accuracy_list = []
    accuracy_list2 = []
    n = X.shape[0]
    it = n//k
    for i in range(k):
        test = X[i*it:(i+1)*it , :]
        train = np.delete(X, slice(i*it, (i+1)*it), 0)
        test_out = Y[i*it:(i+1)*it]
        train_out = np.delete(Y, slice(i*it, (i+1)*it), 0)
        clf = svm.SVC(C = c, kernel = 'rbf', gamma = g)
        clf.fit(train, train_out)
        pred = clf.predict(test)
        pred2 = clf.predict(train)
        accuracy_list.append(metrics.accuracy_score(test_out, pred))
        accuracy_list2.append(metrics.accuracy_score(train_out, pred2))
    return sum(accuracy_list)/len(accuracy_list), sum(accuracy_list2)/len(accuracy_list2)

def kfoldCrossValidationCVX(X, Y, k, c, gamma):
    accuracy_list = []
    accuracy_list2 = []
    n = X.shape[0]
    it = n//k
    for i in range(k):
        test = X[i*it:(i+1)*it , :]
        train = np.delete(X, slice(i*it, (i+1)*it), 0)
        test_out = Y[i*it:(i+1)*it]
        train_out = np.delete(Y, slice(i*it, (i+1)*it), 0)
        w, bias, S = rbfkerneloptimisation(train, train_out, c, gamma)
        h = np.dot(test, w.T) + bias
        for i in range(len(h)):
            if h[i] >= 0:
                h[i] = 1
            else:
                h[i] = -1
        j = np.dot(train, w.T) + bias
        for i in range(len(j)):
            if j[i] >= 0:
                j[i] = 1
            else:
                j[i] = -1
        accuracy_list.append(metrics.accuracy_score(test_out, h))
        accuracy_list2.append(metrics.accuracy_score(train_out, j))
    return sum(accuracy_list)/len(accuracy_list), sum(accuracy_list2)/len(accuracy_list2), S

def linearkerneloptimisation(X, Y, c):
    n = X.shape[0]
    Ym = Y.reshape(1, -1)
    K = np.dot(Ym.T, Ym) * linearkernel(X)
    P = cvxopt.matrix(K)
    q = cvxopt.matrix(-np.ones((n,1)))
    G = cvxopt.matrix(np.vstack((-np.eye((n)), np.eye(n))))
    h = cvxopt.matrix(np.vstack((np.zeros((n,1)), np.ones((n,1)) * c)))
    A = cvxopt.matrix(Ym, (1, n) ,'d')
    b = cvxopt.matrix(np.zeros(1))
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])[:, 0]
    S = (alphas > 1e-5).reshape(-1, )
    w = np.dot(X.T, alphas * Y)
    b = Y[S] - np.dot(X[S], w)
    bias = np.mean(b)
    return w, bias, X[S]

def rbfkerneloptimisation(X, Y, c, g):
    n = X.shape[0]
    Ym = Y.reshape(1, -1)
    K = np.dot(Ym.T, Ym) * rbfkernel(X, g)
    P = cvxopt.matrix(K)
    q = cvxopt.matrix(-np.ones((n,1)))
    G = cvxopt.matrix(np.vstack((-np.eye((n)), np.eye(n))))
    h = cvxopt.matrix(np.vstack((np.zeros((n,1)), np.ones((n,1)) * c)))
    A = cvxopt.matrix(Ym, (1, n) ,'d')
    b = cvxopt.matrix(np.zeros(1))
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])[:, 0]
    S = (alphas > 1e-5).reshape(-1, )
    w = np.dot(X.T, alphas * Y)
    b = Y[S] - np.dot(X[S], w)
    bias = np.mean(b)
    return w, bias, X[S]

def sv_graph(sv, Y, X):
    support_vectors = sv
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(X[:,0], X[:,1], edgecolors=['red' if y_i == -1 else 'blue' for y_i in Y], facecolors='none', s=30)
    ax.scatter(support_vectors[:,0], support_vectors[:,1], c='black', s=50)
    plt.title('CVXOPT')
    plt.show()
    

if __name__ == "__main__":
    f = open("2017EE10500.csv")
    csv_f = csv.reader(f)
    rows = list(csv_f)
    n = len(rows)
    m = (len(rows[0])) - 1
    X = np.zeros([n, m])
    Y = np.arange(n)
    
    for i in range(n):
        for j in range(m):
            X[i][j] = float(rows[i][j])
            
    for i in range(n):
        Y[i] = int(rows[i][25])
    
    #Binary Classification Libsvm
    rows1 = []
    for i in range(n):
        if (int(rows[i][-1]) == 8) or (int(rows[i][-1]) == 9):
            rows1.append(rows[i])
    n1 = len(rows1)
    m1 = (len(rows[1])) - 1
    X1 = np.zeros([n1, m1])
    Y1 = np.arange(n1)
    
    for i in range(n1):
        for j in range(m1):
            X1[i][j] = float(rows1[i][j])
    for i in range(n1):
        Y1[i] = int(rows1[i][25])
    
    #X1 = X1[:, :10]
    
    '''cvalues = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    gamma = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    accuracytest = []
    accuracytrain = []
    for c in cvalues:
        for g in gamma:
            acctest, acctrain = kfoldCrossValidationSVC(X1, Y1, 10, c, g)
            accuracytest.append(acctest)
            accuracytrain.append(acctrain)
    print(accuracytest)
    print(accuracytrain)'''
    
    '''for i in range(n1):
        if Y1[i] == 9:
            Y1[i] = 1
        else:
            Y1[i] = -1
    acctest, acctrain, sv = kfoldCrossValidationCVX(X1, Y1, 10, 100.0, 0.01)
    print(acctest)
    sv_graph(sv, Y1, X1)'''