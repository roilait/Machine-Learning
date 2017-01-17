# -*- coding: utf-8 -*-
import numpy as np
import random, os, sys,  shutil
# IMPORT CLASSES
import cleanData as cld


class Path(object):
    @staticmethod
    def find_file_path():
        # find the file contening the Data that we need
        root_dir = os.path.abspath('../..')
        data_dir = os.path.join(root_dir, 'Machine_Learning')
        return os.path.join(data_dir, 'files/')
        
class DeleteFileOrFolder(object):    
    def __init__(self, directory):
        if os.path.exists(directory):
            try:
                if os.path.isdir(directory):
                    # delete folder
                    shutil.rmtree(directory)
                else:
                    # delete file
                    os.remove(directory)
            except:
                print "Ecxeption ",str(sys.exc_info())
        else:
            print "not found ",directory
               
        
class CvsFileMnist(object):
    @staticmethod
    def manupulations(n_classes):
        train_data = CvsFileMnist.feature_extractor('mnist_train_100', '.csv')
        test_data = CvsFileMnist.feature_extractor('mnist_test_10', '.csv')
        train_y, test_y = [], []
        n_feats = len(train_data[0])-1
        train_x, test_x = np.zeros((len(train_data), n_feats)), np.zeros((len(test_data), n_feats))
        
        #print 'train_x', train_x
        for I in xrange(len(train_data)):
            train_y.append(int(train_data[I][0]))
            train_x[I] = np.array(train_data[I][1:])/255.0 #.append(np.array(train_data[I][387:397]))
            if (I < 10):
                test_y.append(int(test_data[I][0]))
                test_x[I] = np.array(test_data[I][1:])/255.0
        # CONVERT THE INTEGER OUTPUT TO ONE-HOT-ENCODED          
        #train_y = ActFunction.one_hot_encoded(train_y, n_classes)
        #test_y = ActFunction.one_hot_encoded(test_y, n_classes)
        
        
        train_x = []
        train_x.append([1, 1, 1, 1, 1, 1, 0]), train_x.append([0, 1, 1, 0, 0, 0, 0])
        train_x.append([1, 1, 0, 1, 1, 0, 1]), train_x.append([1, 1, 1, 1, 0, 0, 1]) 
        train_x.append([0, 0, 1, 0, 0, 1, 1]), train_x.append([1, 0, 1, 1, 0, 1, 1])
        train_y = train_y[:6]
        train_y = ActFunction.one_hot_encoded(train_y, n_classes) 
        

        # CONVERT THE OUTPUT TO ONE HOT               
        return {'train_x': train_x, 'train_y': train_y, 
                'test_x': None, 'test_y': None,
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
  

         
class Get_needed_data(object):
       
    @staticmethod
    def image_data(feats): # MNIST 
        L = feats['lenght']
        zipfile = cld.UnzipFiles.unzip_files('mnist.pkl.gz')                     
        return { 'train_x': zipfile['trainSet'][0][0:L], 
                 'train_y': zipfile['trainSet'][1][0:L],       
                 'test_x': zipfile['testSet'][0][0:L], 
                 'test_y': zipfile['testSet'][1][0:L],
                 'valid_x': zipfile['validSet'][0][0:L], 
                 'valid_y': zipfile['validSet'][1][0:L],
                 'layers': feats['layers'],
                 'epoches': feats['epochs'],
                 'TrainBatchSize': feats['TrainBatchSize'],        
                 'eta': feats['eta'],
                 'alpha': feats['alpha']
                }                       
    
    
    
    @staticmethod
    def test_date(feats):
        train_x, test_x = list(), list()

        train_x.append([1,1,1,1,1,1,0]), train_x.append([0,1,1,0,0,0,0])
        train_x.append([1,1,0,1,1,0,1]), train_x.append([1,1,1,1,0,0,1]) 
        train_x.append([0,0,1,0,0,1,1]), train_x.append([1,0,1,1,0,1,1])
        
        test_x.append([0,0,1,1,1,1,1]), test_x.append([1,1,1,0,0,0,0])
        
        Set = [1, 0, 0, 1, 0, 1, 0, 0]
        random.shuffle(Set)
        train_y = np.array(Set[0:6])
        test_y = np.array(Set[6:])
        layers = [7, 8, 2]
        
        return { 'train_x': train_x, 'train_y': train_y,       
                 'test_x': test_x,   'test_y': test_y,
                 'valid_x': [],      'valid_y': [],
                 'layers': layers,   'epoches': feats['epochs'],
                 'TrainBatchSize': feats['TrainBatchSize'],        
                 'eta': feats['eta'], 'alpha': feats['alpha']
                }                       
    

    
    @staticmethod        
    def categorized_data(dataFile, feats, percetage):
        # percetage = 0.7
        cleaneData = cld.Clean_data.get_cleaned_data(dataFile, percetage)       
        return { 'train_x': cleaneData['trainSet'],  
                 'train_y': cleaneData['trainValInIndex'],       
                 'test_x': cleaneData['testSet'],       
                 'test_y': cleaneData['testValIndex'],
                 'valid_x': cleaneData['validSet'],      
                 'valid_y': cleaneData['validValIndex'],
                } 
  

              
class EasierData(object):
    
    @staticmethod
    def manupulation(NetData, vectorize, couple_xy = True):
        train_x, train_y = NetData['train_x'], NetData['train_y']
        test_x, test_y = NetData['test_x'], NetData['test_y']
        valid_x, valid_y = NetData['valid_x'], NetData['valid_y']       
        nbrOutput = 10# NetData['layers'][-1]               
        if vectorize:
            train_y = ActFunction.one_hot_encoded(train_y, nbrOutput)
            if test_x!=[]: 
                test_y = ActFunction.one_hot_encoded(test_y, nbrOutput)
            if valid_x!=[]: 
                valid_y = ActFunction.one_hot_encoded(valid_y, nbrOutput)               
        # CHANGE THE DATA AS A COUPLE OF (x,y)        
        if couple_xy:
            Train_data = ActFunction.change_data(train_x, train_y) # [(x,y), (z,t)]
            if test_x!=[]: 
                Test_data = ActFunction.change_data(test_x, test_y)
            if valid_x!=[]: 
                Valid_data = ActFunction.change_data(valid_x, valid_y)
                return Train_data, Test_data, Valid_data
        else:
            return {'train_x': train_x, 
                    'train_y': train_y, 
                    'test_x': test_x, 
                    'test_y': test_y ,
                    'valid_x': valid_x, 
                    'valid_y': valid_y
                    }


        
class ActFunction(object):
    # SIGMOID FUNCTION 
    @staticmethod   
    def sigmoid(z): 
        return 1.0/(1.0 + np.exp(-z))        
    # SIGMOID PRIME FUNCTION     
    @staticmethod
    def sigmoid_prime(pr): 
        return ActFunction.sigmoid(pr)*(1 - ActFunction.sigmoid(pr))
    # VECTORIZE THE OUTPUTS    
    @staticmethod   
    def vectorized(j, nbr):
        binary_output = np.zeros((1, nbr))
        binary_output[0][j] = 1.0
        return binary_output
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