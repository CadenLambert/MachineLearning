# -*- coding: utf-8 -*-
import numpy as np

class KNN:
    def __distance(self, A, B):
        return np.sum((A-B)**2)**0.5
    
    def find(self, X, y, target, k = 7):
        
        num_elem = X.shape[0]
        dist_arr = []
        
        for i in range(num_elem):
            dist = self.__distance(target, X[i])
            dist_arr.append((dist, y[i]))
            
        dist_arr = sorted(dist_arr)
        dist_arr = np.array(dist_arr[:k])
        
        labels = dist_arr[:,1]
        
        u_labl, count = np.unique(labels, return_counts= True)
        pred = int(u_labl[count.argmax()])
        
        return pred