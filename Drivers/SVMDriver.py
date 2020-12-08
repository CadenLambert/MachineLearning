import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SupportVectorMachine import SVM

df = pd.read_csv("../datasets/iris.data", header=None)
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa', -1,1)

"""
0 = sepal length
1 = sepal width
2 = petal length
3 = petal width
"""

X = df.iloc[0:100, [0,3]].values


svm = SVM()
svm.fit(X, y)
 

def hyperplane(x, w, b, offset):
    return (-w[0] * x + b + offset) / w[1]

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o',
label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue',
marker='x', label='versicolor')

x_max = np.amax(X[:,0])
x_min = np.amin(X[:,0])

line_max = hyperplane(x_max, svm.w, svm.b, 0)
line_min = hyperplane(x_min, svm.w, svm.b, 0)
margin1_min = hyperplane(x_min, svm.w, svm.b, -1)
margin1_max = hyperplane(x_max, svm.w, svm.b, -1)
margin2_min = hyperplane(x_min, svm.w, svm.b, 1)
margin2_max = hyperplane(x_max, svm.w, svm.b, 1)


plt.plot([x_min, x_max], [line_min, line_max])
plt.plot([x_min, x_max], [margin1_min, margin1_max], "y--")
plt.plot([x_min, x_max], [margin2_min, margin2_max], "y--")

plt.xlabel("sepal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()