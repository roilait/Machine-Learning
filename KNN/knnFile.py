# -*- coding: utf-8 -*-
''' Par convention, le dernier attribut de la ligne est l'étiquette de la donnée 
    ex: 5.1, 3.5, 1.4, 0.2, Iris-setosa'''
import random, os
from math import sqrt
import numpy as np

# THE KNN CLASSES    
class FilesPath(object):
    @staticmethod
    def path():
        # find the file contening the Data that we need
        root_dir = os.path.abspath('../..')
        data_dir = os.path.join(root_dir, 'Machine_Learning')
        return os.path.join(data_dir, 'files/')
        
class KnnClass(object):  
    # PREPARING THE DATA FOR THE KNN METHOD     
    def __init__(self, dataFilename, extension, k): 
        files = dataFilename+extension
        path = FilesPath.path()
        with open(path + files, 'r') as myfile :
            lines = myfile.readlines()
            Data_set = [l.strip().split(',') for l in lines if l.strip()]            
        random.shuffle(Data_set)
        
        KnnClass.training(Data_set, k)
    @staticmethod    
    def training(Data, k): 
        Train_x, Train_y, Test_x, Test_y  = [], [], [], []          
        for line in xrange(len(Data)):
            for row in xrange(len(Data[line])-1):
                Data[line][row ] = float(Data[line][row])
            # DIVISE THE DATA TO TRAINING AND TESTING DATA     
            if (line<int(len(Data)*0.8)):
                Train_x.append(Data[line][:-1])
                Train_y.append(Data[line][-1])
            else:
                Test_x.append(Data[line][:-1])
                Test_y.append(Data[line][-1])
                
        accuracy = 0.
        for i in range(len(Test_x)):            
            Xnew = Test_x[i]
            y_pred = KnnClass.lesKplusProchesVoisins(Train_x, Train_y, Xnew, k)
            # COMPUTE TBE ACCURACY
            if (y_pred==Test_y[i]):
                accuracy += 1.0/len(Test_x)
            print "The label of this input is: ", Test_y[i], '-', y_pred, '-', accuracy
        
    @staticmethod     
    def lesKplusProchesVoisins(data_x, data_y, X, k):
        # COMPUTE THE DISTANCE BETWEEN Xnew TO THE EXISTING DATA
        print 'X',X
        listOfDistances = [KnnClass.distance(X, data_x[i]) for i in xrange(len(data_x))]             
        # FIND THE K NEAREST NEIGHBOR
        Knn_list = []          
        for i in range(k):
            p = float ("inf")
            for j in range(len(listOfDistances)):
                if (listOfDistances[j]!=0) and (listOfDistances[j]< p) and (j not in Knn_list): 
                    p = listOfDistances[j]
                    indice = j
            Knn_list.append(indice)
            
        # prediction function  
        label_pred = KnnClass.prediction(data_x, data_y, Knn_list)
        
        return label_pred        
    @staticmethod
    def distance(vector1, vector2):
        if (len(vector1) != len(vector2)): 
            return (-1)
        som = np.sum([(a-b)**2 for a, b in zip(vector1, vector2)])        
        return (sqrt(som))
        
    @staticmethod
    def prediction (data_x, data_y, knn_list):
        lesEtiquettes = list(set(data_y)) 
        decomptes = [0]*len(lesEtiquettes)
        for exemple in knn_list :
            for i in range (len(lesEtiquettes)):
                if data_y[exemple] == lesEtiquettes[i]:
                    decomptes[i] += 1
        plusGrandDecompte = decomptes[0]
        indice = 0
        for i in range (1,len(lesEtiquettes)):
            if decomptes[i] > plusGrandDecompte:
                plusGrandDecompte = decomptes[i]
                indice = i
        return (lesEtiquettes[indice])


if __name__=="__main__":  
    # Training phase
    knn_class = KnnClass('iris', '.data', 11)
