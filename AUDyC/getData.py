# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 20:55:00 2017

@author: moussa
"""
import numpy as np
import random, os, sys,  shutil

class Path(object):
    @staticmethod
    def find_file_path():
        # find the file contening the Data that we need
        root_dir = os.path.abspath('../..')
        data_dir = os.path.join(root_dir, 'Machine_Learning')
        return os.path.join(data_dir, 'files/')

        
class CvsFileMnist(object):
    @staticmethod
    def manupulations(n_classes):
        train_data = CvsFileMnist.feature_extractor('mnist_train_100', '.csv')
        test_data = CvsFileMnist.feature_extractor('mnist_test_10', '.csv')
        train_y, test_y = [], []
        n_feats = len(train_data[0])-1
        train_x = np.zeros((len(train_data), n_feats))
        test_x = np.zeros((len(test_data), n_feats))
        
        #print 'train_x', train_x
        for I in xrange(len(train_data)):
            train_y.append(int(train_data[I][0]))
            train_x[I] = np.array(train_data[I][1:])/255.0 #.append(np.array(train_data[I][387:397]))
            if (I < 10):
                test_y.append(int(test_data[I][0]))
                test_x[I] = np.array(test_data[I][1:])/255.0
        # CONVERT THE INTEGER OUTPUT TO ONE-HOT-ENCODED          
        train_y = ActFunction.one_hot_encoded(train_y, n_classes)
        test_y = ActFunction.one_hot_encoded(test_y, n_classes)
        
        '''
        train_x = []
        train_x.append([1, 1, 1, 1, 1, 1, 0]), train_x.append([0, 1, 1, 0, 0, 0, 0])
        train_x.append([1, 1, 0, 1, 1, 0, 1]), train_x.append([1, 1, 1, 1, 0, 0, 1]) 
        train_x.append([0, 0, 1, 0, 0, 1, 1]), train_x.append([1, 0, 1, 1, 0, 1, 1])
        train_y = train_y[:6]
        train_y = ActFunction.one_hot_encoded(train_y, n_classes) 
        '''
        # CONVERT THE OUTPUT TO ONE HOT               
        return {'train_x': train_x, 'train_y': train_y, 
                'test_x': test_x, 'test_y': test_y,
                'valid_x': None, 'valid_y': None
                } 
      
    @staticmethod
    def feature_extractor(fileName, fileExtension):        
        files = fileName + fileExtension 
        # GET THE FILE PATH
        path = Path.find_file_path() 
        with open(path  + files, 'r') as myfile :
            lines = myfile.readlines()
            Data_set = [l.strip().split(',') for l in lines if l.strip()]            
        random.shuffle(Data_set)           
        for line in xrange(len(Data_set)):
            for row in xrange(len(Data_set[line])):
                Data_set[line][row ] = float(Data_set[line][row]) 
                       
        return Data_set  
  
class ActFunction(object):   
    # OUTPUTS ONE HOT ENCODING
    @staticmethod    
    def one_hot_encoded(one_hot, num_class):
        labels_one_hot = np.zeros((len(one_hot),num_class))
        L = [i for i in xrange(len(one_hot))]
        for i, j in zip(L, one_hot): 
            labels_one_hot[i][j] = 1.0
        return labels_one_hot
    # CHANGE THE DATA TO NEW FORMAT    
    @staticmethod
    def change_data(inputs, outputs):
        return [(inputs[i], outputs[i]) for i in xrange(len(inputs))]            