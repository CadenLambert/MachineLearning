# -*- coding: utf-8 -*-

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

