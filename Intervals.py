import numpy as np

class Intervals:
    
    def __init__(self, X):
        self.__dataPoints = X
        self.__bestFunction = None
        self.__allIntervals = None
        
    def allLabelings(self):
        
        allLabels = []
        self.__allIntervals = []
        max_x = max(self.__dataPoints)
        min_x = min(self.__dataPoints)
        
        step = (max_x - min_x)/len(self.__dataPoints)
        
        i = min_x
        max_x += step
        while i <= max_x:
            j = min_x
            while j < i:
                self.__allIntervals.append((j, i))
                j += step
                
            i += step
            
            
        for interval in self.__allIntervals:
            label = np.empty(self.__dataPoints.shape)
            for i in range(len(label)):
                if self.__dataPoints[i] >= interval[0] and self.__dataPoints[i] <= interval[1]:
                    label[i] = 1
                else:
                    label[i] = 0
            allLabels.append(label)
            
        allLabels = np.array(allLabels)
        return allLabels
        
    def setBestFunction(self, x):
        userInterval = self.__allIntervals[x]
        def interval(point):       # define the function with the appropriate bound
            if(point >= userInterval[0] and point <= userInterval[1]):
                return 1
            else:
                return 0
        self.__bestFunction = interval

    def getBestFunction(self):
        return self.__bestFunction