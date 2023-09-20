
import numpy as np
import csv
from sklearn import svm
from sklearn import metrics

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
        
    X = X[:, :10]
        
    cvalues = [0.1, 0.3, 0.6, 1.0, 3.0, 6.0, 10.0]
    gamma = [0.01, 0.03, 0.06, 0.1, 0.13, 0.16, 1.0]
    #degree = [3]
    accuracytest = []
    accuracytrain = []
    maxtest = 0
    #maxdeg = 0
    maxgamma = 0
    maxc = 0
    for c in cvalues:
        for g in gamma:
            #for d in degree:
                acctest, acctrain = kfoldCrossValidationSVC(X, Y, 10, c, g)
                if acctest > maxtest:
                    maxc = c
                    maxgamma = g
                    #maxdeg = d
                    maxtest = acctest
                accuracytest.append(acctest)
                accuracytrain.append(acctrain)
    print(maxc)
    print(maxgamma)
    #print(maxdeg)
    print(maxtest)
    print(accuracytest)
    print(accuracytrain)