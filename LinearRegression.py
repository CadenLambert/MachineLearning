# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
LSR- Least Squares Regression finds the least squares regression function and
MSE- Mean Squared Error and returns both. Takes in an array of independent
variables (x) and an array of dependent variables (y)
"""

def LSR(x,y):
    n = len(x)
    
    m = (n*np.sum(x*y) - np.sum(x)*np.sum(y))/ (n*np.sum(x*x) - np.sum(x)**2)
    b = (np.sum(y) - m*np.sum(x))/n
    
    def lineFunction(x):
        return m*x+b
    
    error = (y - lineFunction(x))
    MSE = np.sum(error*error)/len(error)
    
    return lineFunction, MSE, m, b



df = pd.read_csv("datasets/iris.data", header=None)
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa', -1,1)

"""
0 = sepal length
1 = sepal width
2 = petal length
3 = petal width
"""

X = df.iloc[0:100, [2,3]].values

predictor, Rsquared, m,b = LSR(X[:100, 0], X[:100, 1])
errLabel = "R^2 error: " + str(Rsquared) + "\ny = " + str(m) + "x" + " + " + str(b) 

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o',
label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue',
marker='x', label='versicolor')
plt.plot(X[:100, 0],predictor(X[:100, 0]), label = errLabel)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()
