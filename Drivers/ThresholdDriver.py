import numpy as np
import Threshold as th

test = np.array([1,2,3,4,5,6,7,8,9,10])
thresholdClass = th.Thresholds(test)

print(thresholdClass.allLabelings())

goodInput = False
while goodInput is False:
    choice = int(input("Enter function choice: "))
    
    if choice >= 0 and choice < len(test):
        goodInput = True
        
    else:
        print("Please enter a number between 0 and the length of the array")
thresholdClass.setBestFunction(choice)

bestThreshold = thresholdClass.getBestFunction()

print(bestThreshold(test))

print(bestThreshold(4))




