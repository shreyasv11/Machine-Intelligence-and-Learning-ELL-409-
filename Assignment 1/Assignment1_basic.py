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

def gradientdescent(design_matrix, label_vector, batchsize, learning_rate, iterations):
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
            parameters = parameters - (1/m)*learning_rate*(X.T.dot(h-y))
            cost += leastSquareError(X, y, parameters)
        cost_history[i] = cost
    return parameters, cost_history
    
if __name__ == "__main__":
    f = open("Gaussian_noise.csv")
    csv_f = csv.reader(f)
    rows = list(csv_f)
    
    n = 100
    m = 1
    
    design_matrix = np.zeros([n, m+1])
    
    for i in range(n):
        for j in range(m+1):
            design_matrix[i, j] = pow((float(rows[i][0])),(j))
            
    label_vector = np.zeros([n, 1])
    
    for i in range(n):
        label_vector[i, 0] = float(rows[i][1])
        
        
    """batch = [1, 5, 10, 20, 25, 50]
    newerror = []
    for b in batch:    
        parameters, errlist = gradientdescent(design_matrix, label_vector, b, 0.01, 25000)
        newerror.append(errlist[-1])"""
    
    
    """errorlist = []
    while(m < 15):
        print(m)
        train = design_matrix[ :80, :]
        test = design_matrix[80: , :]
        trainout = label_vector[ :80, :]
        testout = label_vector[80: , :]
        parameters = MoorePenrose(train, trainout)
        error = leastSquareError(design_matrix, label_vector, parameters)
        errorlist.append(error)
        
        m += 1
        design_matrix = np.zeros([n, m+1])
        for i in range(n):
            for j in range(m+1):
                design_matrix[i, j] = pow((float(rows[i][0])),(j))
                
    x = list(range(1, 15))"""