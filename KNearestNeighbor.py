# -*- coding: utf-8 -*-
import numpy as np


"""
KNN is a class that uses the k-nn algorithm to classify a target element.
"""
class KNN:
    
    """
    Helper function to find Euclidian distance of 2 points
    """
    def __distance(self, A, B):
        return np.sum((A-B)**2)**0.5
    
    """
    classify is the implementation of the k-nn algorithm. It tries to classify 
    a target element. X is the data set, y is the labelings for the dataset,
    target is the target element to classify, k is the number of "neighbors" 
    that are checked. 
    """
    def classify(self, X, y, target, k = 7):
        
        #get number of elements
        num_elem = X.shape[0]
        dist_arr = []
        
        #get distance for every element, add it to a list with its label
        for i in range(num_elem):
            dist = self.__distance(target, X[i])
            dist_arr.append((dist, y[i]))
            
        #sort the list to get the elements with the shortest distance at the front
        #and grab k of the closest elements
        dist_arr = sorted(dist_arr)
        dist_arr = np.array(dist_arr[:k])
        
        #get the labels for the k nearest elements
        labels = dist_arr[:,1]
        
        #get a list of unique labels from the k elements and their count
        u_labl, count = np.unique(labels, return_counts= True)
        #pick the element with the highest count
        pred = int(u_labl[count.argmax()])
        
        return pred