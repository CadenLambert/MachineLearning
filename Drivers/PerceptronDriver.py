
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Perceptron import Perceptron as pn
import Perceptron

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa', -1,1)

"""
0 = sepal length
1 = sepal width
2 = petal length
3 = petal width
"""

X = df.iloc[0:100, [2,3]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o',
label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue',
marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()


p = pn(0.1,10)
p.fit(X,y)

print(p.errors)
print(p.weight)

Perceptron.plot_decision_regions(X,y,p)
