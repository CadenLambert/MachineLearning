import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


class Perceptron(object):
    def __init__(self, rate = 0.01, niter = 10):
        self.rate = rate
        self.niter = niter
    def fit(self, X, y):
        """Fit training data
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.shape : [#samples]"""
        # weights: create a weights array of right size
        # and initialize elements to zero
        self.weight = np.zeros([X.shape[1]+1,], dtype=np.float32)
        # Number of misclassifications, creates an array
        # to hold the number of misclassifications
        self.errors = np.zeros([self.niter, ], dtype = int)
        # main loop to fit the data to the labels
        for i in range(self.niter):
            # set iteration error to zero
            iterError = 0
            # loop over all the objects in X and corresponding y element
            for xi, target in zip(X, y):
            # calculate the needed (delta_w) update from previous step
                delta_w = self.rate * (target - self.predict(xi))
            # calculate what the current object will add to the weight
                self.weight[1:] = np.add(self.weight[1:], (delta_w*xi))
            # set the bias to be the current delta_w
                self.weight[0] += delta_w
            # increase the iteration error if delta_w != 0
                if delta_w != 0:
                    iterError += 1
            # Update the misclassification array with # of errors in iteration
            self.errors[i] = iterError
            if(iterError == 0):
                break
                
        return self
     
    def net_input(self, X):
        """Calculate net input"""
        # return the return the dot product: X.w + bias
        return np.dot(X,self.weight[1:]) + self.weight[0]
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa', -1,1)

X = df.iloc[0:100, [2,3]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o',
label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue',
marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()


pn = Perceptron(0.1,10)
pn.fit(X,y)

print(pn.errors)
print(pn.weight)

plot_decision_regions(X,y,pn)

