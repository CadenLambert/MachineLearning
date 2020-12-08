import pandas as pd
import matplotlib.pyplot as plt
import KNearestNeighbor as knn
import numpy as np


train = pd.read_csv("../datasets/mnist_train.csv", header = 0)
test = pd.read_csv("../datasets/mnist_test.csv", header = 0)

y = train.iloc[0:10000, 0].values
X = train.iloc[0:10000, 1:].values

X_test = test.iloc[0:1000, 1:].values
y_test = test.iloc[0:1000, 0].values

kNN = knn.KNN()

# Must be with within range of the test values
index = 500 

result = kNN.find(X, y, X_test[index])

print("Found result: " + str(result))
print("Actual label: " + str(y_test[index]))

pred = []
for i in range(100):
    temp = kNN.find(X, y, X_test[i])
    pred.append(temp)
    
pred = np.array(pred)

accuracy = (y_test[:100] == pred).sum()/len(pred)

print("Prediction accuracy: " + str(accuracy))