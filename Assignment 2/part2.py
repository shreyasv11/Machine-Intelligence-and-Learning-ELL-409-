
import numpy as np
import csv
from sklearn import svm
from sklearn import metrics

def kfoldCrossValidationSVC(X, Y, k, c, g, d):
    accuracy_list = []
    accuracy_list2 = []
    n = X.shape[0]
    it = n//k
    for i in range(k):
        test = X[i*it:(i+1)*it , :]
        train = np.delete(X, slice(i*it, (i+1)*it), 0)
        test_out = Y[i*it:(i+1)*it]
        train_out = np.delete(Y, slice(i*it, (i+1)*it), 0)
        clf = svm.SVC(C = c, kernel = 'poly', gamma = g, degree = d)
        clf.fit(train, train_out)
        pred = clf.predict(test)
        pred2 = clf.predict(train)
        accuracy_list.append(metrics.accuracy_score(test_out, pred))
        accuracy_list2.append(metrics.accuracy_score(train_out, pred2))
    return sum(accuracy_list)/len(accuracy_list), sum(accuracy_list2)/len(accuracy_list2)



if __name__ == "__main__":
    f = open("train_set.csv")
    csv_f = csv.reader(f)
    rows = list(csv_f)
    n = len(rows)
    m = (len(rows[0])) - 1
    X = np.zeros([n, m])
    Y = np.arange(n)
    
    for i in range(n):
        for j in range(m):
            X[i][j] = float(rows[i][j])
            
    for j in range(m):
        t = X[:, j]
        maxi = np.max(t)
        mini = np.min(t)
        length = maxi - mini
        X[:, j] = ((X[:, j]) - mini)/length
            
    for i in range(n):
        Y[i] = int(rows[i][25])
        
    cvalues = [0.05, 0.1, 0.15]
    gamma = [0.15, 1.0, 2.0]
    degree = [4, 5]
    accuracytest = []
    accuracytrain = []
    maxtest = 0
    maxdeg = 0
    maxgamma = 0
    maxc = 0
    for c in cvalues:
        for g in gamma:
            for d in degree:
                acctest, acctrain = kfoldCrossValidationSVC(X, Y, 10, c, g, d)
                if acctest > maxtest:
                    maxc = c
                    maxgamma = g
                    maxdeg = d
                    maxtest = acctest
                accuracytest.append(acctest)
                accuracytrain.append(acctrain)
    print(maxc)
    print(maxgamma)
    print(maxdeg)
    print(maxtest)
    
    '''g = open("test_set.csv")
    csv_g = csv.reader(g)
    rows1 = list(csv_g)
    n = len(rows1)
    m = (len(rows1[0]))
    testX = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            testX[i][j] = float(rows1[i][j])       
    for j in range(m):
        t = testX[:, j]
        maxi = np.max(t)
        mini = np.min(t)
        length = maxi - mini
        testX[:, j] = ((testX[:, j]) - mini)/length
    
    clf = svm.SVC(C = 4.0, kernel = 'rbf', gamma = 3.0)
    clf.fit(X, Y)
    pred = clf.predict(testX)
    predlist = []
    for i in range(n):
        predlist.append(pred[i])
        
    ids = [a for a in range(2000)]
    
    with open('test.csv', mode='w', newline = '') as t:
        t_writer = csv.writer(t, delimiter = ',')
        for i in range(n):
            t_writer.writerow([ids[i], predlist[i]])'''