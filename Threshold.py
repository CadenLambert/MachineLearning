# -*- coding: utf-8 -*-
import numpy as np

"""
The Thresholds class takes in a set of data points and by calling the 
allLabeling function, will produce a 2d array of all the labelings where each
row is a different threshold function. You can then set the best function by 
inputting the index of the desired function and that function will be saved.
You can then call the getBestFunction function and that function will be 
returned
"""

class Thresholds:
    
    def __init__(self, X):
        self.__dataPoints = X
        self.__bestFunction = None
        
    def allLabelings(self):
        allLabels = np.empty([len(self.__dataPoints), len(self.__dataPoints)])
        count = 0
        for i in self.__dataPoints:
            y = np.where(self.__dataPoints < i, 0, 1)   # sets everything in less than the i to 0, 1 otherwise
            allLabels[count] = y
            count += 1
        return allLabels
                
    def setBestFunction(self, x):
        bound = self.__dataPoints[x]
        def threshold(point):       # define the function with the appropriate bound
            return np.where(point < bound, 0, 1)
        self.__bestFunction = threshold

    def getBestFunction(self):
        return self.__bestFunction
        

test = np.array([1,2,3,4,5,6,7,8,9,10])
hypothesisClass = Thresholds(test)

print(hypothesisClass.allLabelings())

goodInput = False
while goodInput is False:
    choice = int(input("Enter function choice: "))
    
    if choice >= 0 and choice < len(test):
        goodInput = True
        
    else:
        print("Please enter a number between 0 and the length of the array")
hypothesisClass.setBestFunction(choice)

bestThreshold = hypothesisClass.getBestFunction()

print(bestThreshold(test))

print(bestThreshold(4))
