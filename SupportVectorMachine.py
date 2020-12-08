import numpy as np

class SVM:

    def __init__(self, learning_rate=0.01, epochs=10000):
        self.learning_rate = learning_rate
        self.lmda = 1/ epochs
        self.epochs = epochs
        self.w = None
        self.b = None


    def fit(self, X, y):      
        self.w = np.zeros(X[1].shape)
        self.b = 0

        for epoch in range(self.epochs):
            for i in range(len(X)):
                condition = y[i] * (np.dot(X[i], self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lmda * self.w)
                    #b -= learning_rate * 0
                else:
                    self.w -= self.learning_rate * (2 * self.lmda * self.w - np.dot(X[i], y[i]))
                    self.b -= self.learning_rate * y[i]


    def predict(self, X):
        pred = np.dot(X, self.w) - self.b
        return np.sign(pred)