import numpy as np
import Intervals as vl

test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

intervalClass = vl.Intervals(test)

Labelings = intervalClass.allLabelings()

count = 0
for labels in Labelings:
    print(str(count)+". " + str(labels))
    count += 1

choice = int(input("Enter function choice: "))
    
        
intervalClass.setBestFunction(choice)

bestInterval = intervalClass.getBestFunction()

for i in test:
    print(bestInterval(i))
    
    
print("\n" + str(bestInterval(6)))