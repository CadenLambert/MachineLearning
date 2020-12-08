# -*- coding: utf-8 -*-
import numpy as np

"""
LR is a class that uses a sigmoid function and gradient descent for optimization
in finding logistic regression.
"""
class LR:
    
    def __init__(self, learning_rate = 0.01, epochs = 10000):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.theta = None
        self.loss = None
        
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    
    """
    train takes in a dataset and its labels (X and y respectfully. y assumes
    0,1 labeling). Uses gradient descent to optimize the weights. Find a z by 
    taking the dot product of datasets features and the weights(theta). Plug z
    into the sigmoid function. Find the gradient. The gradient is the dot product
    of the features of the dataset and h minus every label all divided by the
    number of elements. Then update the weights by subtracting the 
    learning rate * gradient off all weights. Repeat those steps for the given
    number of epochs. Also store the loss for each epoch in a list
    """
    def train(self, X, y):
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        self.loss = []
        
        #gradient descent
        for i in range(self.epochs):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_rate * gradient 
            
            self.loss.append(self.__loss(h, y))
            
    """
    predicts the probability that element is of label 1
    """
    def predict_prob(self, X):
        return self.__sigmoid(np.dot(X, self.theta))
    
    """
    uses predict_prob and a threshold to produce a labeling 1 or 0
    """
    def predict(self, X, threshold = 0.5):
        return self.predict_prob(X) >= threshold
    
    
    