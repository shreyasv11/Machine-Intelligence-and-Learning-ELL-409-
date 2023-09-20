

import numpy as np
import csv
from matplotlib import pyplot as plt

def leastSquareError(design_matrix, label_vector, parameters):
    h = np.dot(design_matrix, parameters)
    h = np.subtract(h, label_vector)
    err = np.sum(np.square(h))
    return (err)/design_matrix.shape[0]

def regularizedleastSquareError(design_matrix, label_vector, parameters, Lambda):
    m = design_matrix.shape[0]
    J = 0
    h = np.dot(design_matrix,parameters)
    J_reg = (Lambda/(2*m))*np.sum(np.square(parameters))
    J = float((1./(2*m)) * (np.dot((h - label_vector).T,(h - label_vector)))) + J_reg;
    return(J) 

def regularizedMoorePenrose(design_matrix, label_vector, rate):
    a = np.dot(design_matrix.transpose(),design_matrix)
    i = np.identity(a.shape[0])
    i = rate * i
    a = a + i
    ainv = np.linalg.inv(a)
    b = np.dot(ainv, design_matrix.transpose())
    parameters = np.dot(b, label_vector)
    return parameters

def regularizedgradientdescent(design_matrix, label_vector, batchsize, learning_rate, iterations, Lambda):
    m = design_matrix.shape[0]
    n = design_matrix.shape[1]
    parameters = np.zeros([n, 1])
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        cost = 0.0
        indices = np.random.permutation(m)
        design_matrix = design_matrix[indices]
        label_vector = label_vector[indices]
        for j in range(0, m, batchsize):
            X = design_matrix[j:j+batchsize]
            y = label_vector[j:j+batchsize]
            h = np.dot(X, parameters)
            parameters = parameters - (1/m)*learning_rate*(X.T.dot(h-y) + Lambda*parameters)
            cost += regularizedleastSquareError(X, y, parameters, Lambda)
        cost_history[i] = cost
    return parameters, cost_history

def kfoldCrossValidation(design_matrix, label_vector, Lambda):
    error_list = []
    for i in range(10):
        test = design_matrix[i*10:(i+1)*10 , :]
        train = np.delete(design_matrix, slice(i*10, (i+1)*10), 0)
        test_out = label_vector[i*10:(i+1)*10 , :]
        train_out = np.delete(label_vector, slice(i*10, (i+1)*10), 0)
        parameters, c = regularizedgradientdescent(train, train_out, 10, 0.001, 5000, Lambda)
        error_list.append(leastSquareError(test, test_out, parameters))
    return sum(error_list)/len(error_list)
        
    
if __name__ == "__main__":
    f = open("Gaussian_noise.csv")
    csv_f = csv.reader(f)
    rows = list(csv_f)
    
    n = 100
    m = 12
    
    design_matrix = np.zeros([n, m+1])
    
    for i in range(n):
        for j in range(m+1):
            design_matrix[i, j] = pow((float(rows[i][0])),(j))
            
    label_vector = np.zeros([n, 1])
    
    for i in range(n):
        label_vector[i, 0] = float(rows[i][1])
    
    """lamda = [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
    costlist = []
    for l in lamda:
        costlist.append(kfoldCrossValidation(design_matrix, label_vector, l))"""
        
    parameters = regularizedMoorePenrose(design_matrix, label_vector, 0.0003)
    ypred = np.dot(design_matrix, parameters)
    x = design_matrix[:, 1]
    #plt.scatter(x, ypred)
    error = label_vector - ypred
    plt.scatter(x, error)
    su = np.sum(error)