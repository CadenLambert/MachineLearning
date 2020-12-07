import numpy as np
import Threshold as th

test = np.array([1,2,3,4,5,6,7,8,9,10])
hypothesisClass = th.Thresholds(test)

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