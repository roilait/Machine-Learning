# -*- coding: utf-8 -*-

# https://www.youtube.com/watch?v=RD0nNK51Fp8
# https://www.youtube.com/watch?v=Ro9EpTzFvoI
# https://www.youtube.com/watch?v=0MQEt10e4NM
# https://www.youtube.com/watch?v=rjm4slbER_M
"""
Created on Tue Jan  3 15:04:38 2017
@author: moussa
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random, os
import numpy as np
from math import sqrt

# THE KNN CLASSES    
class FilesPath(object):
    @staticmethod
    def path():
        # find the file contening the Data that we need
        root_dir = os.path.abspath('../..')
        data_dir = os.path.join(root_dir, 'Machine_Learning')
        return os.path.join(data_dir, 'files/')
        
class KMeansClass(object):
    def __init__(self, dataFilename, extension, n_class):
        files = dataFilename+extension
        path = FilesPath.path()
        with open(path + files, 'r') as myfile :
            lines = myfile.readlines()
            Data_set = [l.strip().split(',') for l in lines if l.strip()]            
        random.shuffle(Data_set)
        
        # KMEANS CLUSTERING 
        classes, centroids, avg_dist = KMeansClass.picking_right_k(Data_set)
        # KMEANS PREDICTION
        # KMeansClass.kMeans_predict(classesData, centroids, Xnew)
        # VISUALIZE THE DATA
        KMeansClass.visualize_data(classes, centroids)
    
    @staticmethod
    def picking_right_k(Data):
        K, flag = 1, False
        avg_dist, K_list = [], []
        while K<5: # not flag:
            # CLUSTERING WITH THE VALUE OF K
            centroids, classesData = KMeansClass.kMeans_clustering(Data, K) 
            # COMPUTE THE AVERAGE DISTANCE TO CENTROID
            dist = 0.
            for I in range(len(centroids)):
                # DISTANCE BETWEEN THE CENTROID I TO DATE IN THE CLASS I
                dist_list = [KMeansClass.distance(centroids[I], classesData['C'+str(I)][j]) 
                                                          for j in xrange(len(classesData['C'+str(I)]))]
                dist += np.mean(dist_list) 
            # avg_dist IS USED TO CHOOCE THE RIGHT K
            avg_dist.append(dist/K)
            K_list.append(K)
            K += 1             
       
        return classesData, centroids, avg_dist
        
    @staticmethod
    def kMeans_predict(classesData, centroids, Xnew):
        computDist = [KMeansClass.distance(Xnew, centroids[I]) for I in xrange(len(centroids))]
        classIndex = computDist.index(min(computDist))
        classesData['C'+str(classIndex)].append(Xnew)
        
        return "The new data 'Xnew' belong to the class,", 'C'+str(classIndex) 
        
    @staticmethod
    def visualize_data(classes, centroids):
        col = ['g', 'r', 'c', 'm', 'y', 'k', 'w','r', 'c', 'm', 'y', 'k', 'w']
        if (len(centroids[0])==2):            
            plt.figure()
            for k in range(len(centroids)):
                x = [classes['C'+str(k)][i][0] for i in xrange(len(classes['C'+str(k)]))]
                y = [classes['C'+str(k)][j][1] for j in xrange(len(classes['C'+str(k)]))]
                plt.plot(x,y, col[k], marker='o')
                plt.plot(centroids[k][0], centroids[k][1], c='b', marker='o')
            plt.show()
        elif (len(centroids[0])==3):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for cr in range(len(centroids)):
                x = [classes['C'+str(cr)][i][0] for i in xrange(len(classes['C'+str(cr)]))]
                y = [classes['C'+str(cr)][j][1] for j in xrange(len(classes['C'+str(cr)]))]
                z = [classes['C'+str(cr)][k][2] for k in xrange(len(classes['C'+str(cr)]))]
                ax.scatter(x, y, z, c=col[cr], marker='o')
                ax.scatter(centroids[cr][0], centroids[cr][1], centroids[cr][2], c='b', marker='o')
            plt.show()
        else: print 'Dim of the Data is higher than 3!'
            
    @staticmethod    
    def kMeans_clustering(Data, n_classes):
        # ------------------ Begin ---------------------------
        train_x, test_x = [], [] 
        # train_y, test_y  = [], [], [], []          
        for line in xrange(len(Data)):
            for row in xrange(len(Data[line])-1): 
                Data[line][row ] = float(Data[line][row])
            # DIVISE THE DATA TO TRAINING AND TESTING DATA    
            if (line<int(len(Data)*1)):
                train_x.append(Data[line][:-1])
                # train_y.append(Data[line][-1])
            else:
                test_x.append(Data[line][:-1])
                # test_y.append(Data[line][-1])
        # ---------------------- End ----------------------------------
                
        # 1 - INITIALIZATION CENTROIDS
        centroids = KMeansClass.init_centroid(train_x, n_classes)
        # STARD THE CLUSTERING
        moveGroup, stop_test = True, []
        while moveGroup:
            # CLASSES INITIALIZATION, THE DATA WILL BE AFFECTED TO THE CLASSES
            classesData = {'C'+str(I):[centroids[I]] for I in xrange(len(centroids))}
            # 2 - CHOOSE OPTIMAL A FOR FIXED CENTROIDS
            for obs in train_x:
                computDist = [KMeansClass.distance(obs, centroids[I]) for I in xrange(len(centroids))]
                classIndex = computDist.index(min(computDist))
                classesData['C'+str(classIndex)].append(obs) 
            # 3 - CHOISE OPTIMAL CENTROIDS FOR FIXED A
            centroids = KMeansClass.update_centroid(classesData, centroids)
            # 4 - NO OBJECT MOVE OBJECT ?
            if not stop_test:
                stop_test = [len(classesData['C'+str(I)]) for I in xrange(len(centroids))]
            else:
                for itera in xrange(len(centroids)):
                    moveGroup = False # STABILITY IS DETECTED
                    # p, L = stop_test[itera][0], stop_test[itera][1]
                    if stop_test[itera]!=len(classesData['C'+str(itera)]):
                        stop_test = [len(classesData['C'+str(k)]) for k in xrange(len(centroids))]
                        moveGroup = True
                        print 'encore'
                        break  
                    
        return centroids, classesData
        
    @staticmethod
    def update_centroid(Data, centroid):
        return [((np.mean(Data['C'+str(cls)], axis=0)).tolist()) for cls in xrange(len(centroid))]
    
    @staticmethod
    def init_centroid(data, n_class):
        centroids = [random.choice(data) for i in xrange(n_class)]
        # HERE WE CHECK IF WE DID NOT PICKED RANDOMLY SAME CENTROID
        redundancy = True
        while redundancy:
            redundancy = False
            for cent in xrange(len(centroids)):
                theCurr = centroids[cent]
                theRest = [centroids[item] for item in xrange(len(centroids)) if item!=cent]
                if theCurr in theRest:
                    centroids = [random.choice(data) for i in xrange(n_class)]
                    redundancy = True
                    break 
                
        return centroids
        
    @staticmethod
    def distance(vector1, vector2):
        if (len(vector1) != len(vector2)): 
            return (-1)
        dist = sqrt(np.sum([(a-b)**2 for a, b in zip(vector1, vector2)]))        
        return dist
      
if __name__=="__main__":
    fileName, fileExtension = 'iris', '.data'
    fileName, fileExtension = 'regre1', '.data'
    n_classes =7
    # UNSUPERVISED METHOD, K-MEANS ALGORITHM  
    kncls = KMeansClass(fileName, fileExtension, n_classes)


            
        
