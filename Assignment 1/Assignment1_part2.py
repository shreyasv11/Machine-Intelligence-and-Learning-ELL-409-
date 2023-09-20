

import numpy as np
import csv
from matplotlib import pyplot as plt

def leastSquareError(design_matrix, label_vector, parameters):
    h = np.dot(design_matrix, parameters)
    h = np.subtract(h, label_vector)
    err = np.sum(np.square(h))
    return (err)/design_matrix.shape[0]

def MoorePenrose(design_matrix, label_vector):
    a = np.dot(design_matrix.transpose(),design_matrix)
    ainv = np.linalg.inv(a)
    b = np.dot(ainv, design_matrix.transpose())
    parameters = np.dot(b, label_vector)
    return parameters

def kfoldCrossValidation(design_matrix, label_vector):
    error_list = []
    for i in range(11):
        test = design_matrix[i*10:(i+1)*10 , :]
        train = np.delete(design_matrix, slice(i*10, (i+1)*10), 0)
        test_out = label_vector[i*10:(i+1)*10 , :]
        train_out = np.delete(label_vector, slice(i*10, (i+1)*10), 0)
        parameters = MoorePenrose(train, train_out)
        error_list.append(leastSquareError(test, test_out, parameters))
    return sum(error_list)/len(error_list)


if __name__ == "__main__":
    f = open("train.csv")
    csv_f = csv.reader(f)
    rows = list(csv_f)
    n = len(rows)
    dates = []
    values = []
    for i in range(1, n):
        month, day, year = rows[i][0].split("/")
        dates.append((int(month), int(year)))
        values.append(float(rows[i][1]))
    n = n-1
    x = []
    y = []
    for i in range(n):
        month, year = dates[i]
        x.append(month)
        y.append(values[i])
    
    m = 1
    n = 110
    
    design_matrix = np.zeros([n, m+1])
    for i in range(n):
        for j in range(m+1):
            design_matrix[i, j] = pow((float(x[i])),(j))
            
    label_vector = np.zeros([n, 1])
    for i in range(n):
        label_vector[i, 0] = float(y[i])
        
    errlist = []     
    while(m<15):
        err = kfoldCrossValidation(design_matrix, label_vector)
        errlist.append(err)
        m += 1
        design_matrix = np.zeros([n, m+1])
        for i in range(n):
            for j in range(m+1):
                design_matrix[i, j] = pow((float(x[i])),(j))
                
    m = 6
    design_matrix = np.zeros([n, m+1])
    for i in range(n):
        for j in range(m+1):
            design_matrix[i, j] = pow((float(x[i])),(j))
            
    parameters = MoorePenrose(design_matrix, label_vector)
    
    g = open("test.csv")
    csv_g = csv.reader(g)
    rowg = list(csv_g)
    testdata = np.zeros([10, 1])
    
    for k in range(1, 11):
        month, day, year = rowg[k][0].split("/")
        testdata[k-1] = int(month)
        
    design_matrix_test = np.zeros([10, m+1])
    for i in range(10):
        for j in range(m+1):
            design_matrix_test[i, j] = pow((float(testdata[i])),(j))
    
    ypred = np.dot(design_matrix_test, parameters)
        