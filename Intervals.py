import numpy as np

"""
Intervals is a hypothesis class that takes in a 1-d numpy array of data and finds
all possible intervals within that data. 
"""
class Intervals:
    
    def __init__(self, X):
        self.__dataPoints = X
        self.__bestFunction = None
        self.__allIntervals = None
        
    """
    The allLabelings function returns all of the labels possible from the
    given dataset.  The intervals are determined by finding the min and max of
    the dataset and a step size found by subtracting the min from the max divided
    by the length of the dataset. Now it loops through all possible intervals 
    by starting at the min and incrementing by the step size for the outer loop 
    and doing the same for the inner loop. Once the intervals have been generated
    it loops through all of them and labels the points in the dataset accordingly
    and then add that labeling to an array. Finally it returns the array of labels
    """ 
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
        
    """
    setBestFunction takes in the index of the desired labeling and defines a
    function that has that definition.
    """
    def setBestFunction(self, x):
        userInterval = self.__allIntervals[x]
        def interval(point):       # define the function with the appropriate bound
            if(point >= userInterval[0] and point <= userInterval[1]):
                return 1
            else:
                return 0
        self.__bestFunction = interval
        

    """
    getBestFunction returns the function defined in setBestFunction.
    """
    def getBestFunction(self):
        return self.__bestFunction