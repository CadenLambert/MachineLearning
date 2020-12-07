# -*- coding: utf-8 -*-
import numpy as np

class LR:
    
    def __init__(self, learning_rate = 0.01, epochs = 10000):
        self.epochs = epochs
        self.learning_rate = learning_rate
        
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def train(self, X, y):
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        self.loss = []
        for i in range(self.epochs):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_rate * gradient
            
            self.loss.append(self.__loss(h, y))
    
    def predict_prob(self, X):
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold = 0.5):
        return self.predict_prob(X) >= threshold
    
    
    