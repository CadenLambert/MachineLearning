import numpy as np


"""
SVM is a class that implements the support vector machine. It takes
in a learning rate and number of epochs and sets the regularization parameter 
(lambda) to 1/epochs.
"""

class SVM:

    def __init__(self, learning_rate=0.01, epochs=10000):
        self.learning_rate = learning_rate
        self.lmda = 1/epochs
        self.epochs = epochs
        self.w = None
        self.b = None


    """
    fit takes in a dataset and its labels (X and y respectively. y assumes -1,1
    labeling). First the weights and intercept are initialized to 0. Next it 
    begins a loop through all the epochs for gradient descent. In this loop,
    it loops over all the elements in the dataset and checks if it classifies 
    it correctly. if so, then update the weights by taking the learning rate
    and multiply it by 2*lambda*weights (gradient) and subtract the result from
    the current weights. If it isn't classified correctly, then update the weights
    by using taking the same gradient and subtracting the dot product of the 
    features of the current element and the label, multiplying the result by 
    the learning rate and subtracting the results from the weights. The 
    intercept is also updated by subtracting the learning rate multiplied by 
    the label from the current intercept. This process is repeated for all epochs.
    """
    def fit(self, X, y):      
        self.w = np.zeros(X[1].shape)
        self.b = 0

        for epoch in range(self.epochs):
            for i in range(len(X)):
                correct = y[i] * (np.dot(X[i], self.w) - self.b) >= 1
                if correct:
                    self.w -= self.learning_rate * (2 * self.lmda * self.w)
                    #b -= learning_rate * 0
                else:
                    self.w -= self.learning_rate * (2 * self.lmda * self.w - np.dot(X[i], y[i]))
                    self.b -= self.learning_rate * y[i]

    """
    predict takes the dot product of the features with the weights and subtracts
    the intercept. Returns 1 if the sign of the result is positive, -1 otherwise
    """
    def predict(self, X):
        pred = np.dot(X, self.w) - self.b
        return np.sign(pred)