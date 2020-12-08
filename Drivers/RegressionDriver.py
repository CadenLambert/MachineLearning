import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import LinearRegression
import LogisticRegression

df = pd.read_csv("../datasets/iris.data", header=None)
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa', 0,1)

"""
0 = sepal length
1 = sepal width
2 = petal length
3 = petal width
"""

X = df.iloc[0:100, [2,3]].values
plt.title("Linear Regression")
predictor, Rsquared, m,b = LinearRegression.LSR(X[:100, 0], X[:100, 1])
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

log = LogisticRegression.LR()
log.train(X, y)
data = log.loss

plt.title("Logistic Regression Error")
plt.plot(range(0,len(data)), data)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()