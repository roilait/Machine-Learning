# -*- coding: utf-8 -*-
#import NearestNeighbor as knn
import numpy as np
class Descriptor(object):
    def __init__(inputs, n_class, ro, normalizer=True):
        if normalizer:
            MAD = [[ro**x_id(1-ro)**(1-x_id) for x_id in inputs] for c in n_class]
        else:
            Max, Min = max(inputs), min(inputs)
            MAD = [[ro**((x_id-Min)/(Max-Min))(1-ro)**(1-((x_id-Min)/(Max-Min))) for x_id in inputs] for c in n_class]
            
        
        
if __name__=="__main__":  
    # Training phase
    #knn_class = knn.KnnClass('iris', '.data', 11)
    a = [3,5,2, 3, 6,8]
    b = {'c'+str(i):np.zeros((a[i],a[i+1])) for i in range(len(a)-1)}
    
    print 'c0: ', len(b)

