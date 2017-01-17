# -*- coding: utf-8 -*-
# import matplotlib.pyplot as plt
from __future__ import division
import numpy as np
import json, copy
# THE CLASSES IMPORTED
import cleanData as cld
import getData as getd

class DataForTraining(object):
    def __init__(self, NetData, vectorize = True):
        Train_data, Test_data, Valid_data = getd.EasierData.manupulation(NetData, vectorize)
        eta, alpha = NetData['eta'], NetData['alpha']
        epoches, layers = NetData['epoches'], NetData['layers']
        mbs = NetData['miniBatchSize']              
                
        # -----------------------
        # call the training function
        # -----------------------
        # self.train_network(layers)
        
        self.cost, self.accuracy = Training.train_network(layers, epochs, eta,
                                            alpha, Train_data, Test_data,mbs)   
    def getNetOutput(self):
        return self.cost, self.accuracy


        
class Training(object):
    @staticmethod
    def train_network(layers, epoches, eta, alpha, Train_data, Test_data,mbs):
        # THE NEURAL NETWORK CLASS 
        Ann = NeuralNetwork(layers)
        n = len(Train_data)
        # ------------------------        
        epoch, training_cost, test_cost, test_accuracy = 0, list(), list(), list()    
        while epoch < epoches:
            # TRAINING PHASE
            batches = [Train_data[j:j + mbs] for j in xrange(0, n, mbs)]
            # tr_avg_cost = []
            cost = [Ann.update_params(mini_batches, eta, alpha) for mini_batches in batches]
            # cost = [Ann.update_params(x,y,eta,alpha) for i in range(len(Train_data)) for x, y in [Train_data[i]] ]  
            training_cost.append(np.mean(cost))            
            # EVALUATION PHASE             
            if Test_data:
                accuracy, cost = Ann.net_evalution(Test_data)
                test_cost.append(cost)
                test_accuracy.append(accuracy)
                print "Epoch {0}: {1}".format(epoch, accuracy) 
                
            # -------- SAVE NEURAL NETWORK HYPER-PARAMETERS -------------
            hyperParams = Ann.getHyperParams()                           
            fileName = cld.FileDirectory.find_file_path() + 'Netparams'
            saveNet = { 'size':hyperParams['layers'],
                        'epoch': epoch,
                        'weights':hyperParams['weigts'],
                        'biases':hyperParams['biases'],
                        'cost': 0 }
            if epoch==0:
               LoadingNetwork.saveNetwork(saveNet, fileName) 
            # -----------------------------------------------------------------
               
            epoch += 1
        cost = {'train': training_cost, 'test': test_cost}
        accuracy = {'train': test_accuracy}
        
        return cost, accuracy



class NeuralNetwork(object):    
    def __init__(self, layers):
        self.layers, self.num_layers = layers, len(layers), 
        self.biases = [np.random.randn(k, 1) for k in layers[1:]]
        self.weights = [np.random.rand(x, k) for x,k in zip(layers[:-1], layers[1:])]
        
        
    # FORWARD PROPAGATION FUNCTION     
    def forward_prop(self, x):        
        activation = np.array([x])  
        # list to store all the z vectors, layer by layer  
        # list to store all the activations, layer by layer
        self.noActiv, self.matrix = [], {'forward': []}
        self.matrix['forward'].append(activation)
        for w, b in zip(self.weights, self.biases):
            f = np.dot(activation, w) + b.T
            self.noActiv.append(f)
            activation = getd.ActFunction.sigmoid(f)
            self.matrix['forward'].append(activation)        
        return self.matrix['forward'][-1]    
   
    # BACKWARD PROPAGATION FUNCTION    
    def back_propa(self, inputs, truthVal):
        # THE FORWARD PROPAGATION FUNCTION IS CALLED
        predicts = self.forward_prop(inputs)
        # ------------------------------------------------
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        x, targets = self.noActiv[-1], truthVal        
        avg_cost = QuadratiCost.mse(targets, predicts)        
        delta = QuadratiCost.delta(x, targets, predicts)
        #avg_cost = CrossEntropyCost.cross_cost(targets, predicts)
        #delta = CrossEntropyCost.delta(targets, predicts).T
        nabla_b[-1] = delta
        nabla_w[-1] = (np.dot(delta, self.matrix['forward'][-2])).T        
        for i in xrange(2, self.num_layers):
            k = self.noActiv[-i]            
            back = getd.ActFunction.sigmoid_prime(k).T  
            delta = np.dot(self.weights[-i+1], delta)*back
            nabla_b[-i] = delta
            nabla_w[-i] = (np.dot(delta, self.matrix['forward'][-i-1])).T             
        return avg_cost, nabla_w, nabla_b  
        
    # UPDATE PARAMETERS FUNCTION 
    def update_params(self, mini_batches, eta, alpha):          
        # THE BACKWARD PROPAGATION FUNCTION IS CALLED
        avg_cost = []
        for x, y in mini_batches:
            cost, delta_nabla_w, delta_nabla_b = self.back_propa(x, y) 
            self.weights = [w + alpha*w + eta*nw for w, nw in zip(self.weights, delta_nabla_w)] 
            self.biases = [b + alpha*b + eta*nb for b, nb in zip(self.biases, delta_nabla_b)]
            avg_cost.append(cost)
        return np.mean(avg_cost)
        
    def net_evalution(self, test_data):
        # cost = QuadratiCost.mse(targets, predicts)        
        # cost = CrossEntropyCost.cross_cost(targets, predicts)
        results = [(np.argmax(self.forward_prop(x)), np.argmax(y)) for (x,y) in test_data]
        cost = np.mean([np.mean(QuadratiCost.mse(y, self.forward_prop(x))) for x,y in test_data])
        
        return round(sum(int(x==y) for x,y in results)/(len(test_data)),3), cost
        
    def getHyperParams(self):
        return { 'layers': self.layers,
                 'biases': self.biases, 
                 'weigts': self.weights
               } 

                           
# THIS CLASS RETURN A INSTANCE OF THE NETWORK  
class LoadingNetwork (object):
    @staticmethod
    def saveNetwork(saveNet, fileName):
        data = { 'size': saveNet['size'],
                 'epoch': saveNet['epoch'],
                 'weights':[w.tolist() for w in saveNet['weights']],
                 'biases':[b.tolist() for b in saveNet['biases']],
                 'cost': saveNet['cost']
                }                
        files = open(fileName, "w")
        json.dump(data, files)
        files.close()


        
class QuadratiCost(object):
    @staticmethod
    def mse(targets, predicts): # MEAN SQUARED ERROR  
        return np.mean((targets-predicts)**2) # RETURN THE AVERAGE COST
    @staticmethod
    def mse_deriv(targets, predicts):
        return targets-predicts 
    @staticmethod
    def delta(inputs, targets, predicts):
        z_prime = getd.ActFunction.sigmoid_prime(inputs).T
        return z_prime*(targets-predicts).T


       
class CrossEntropyCost(object):
    # THE np.nan_to_num FUNCTION ENSURES THAT np.log(0) EVALUATE TO 0 INSTEAD OF NAN
    @staticmethod    
    def cross_cost(targets, predicts):
        return -np.nan_to_num(targets*np.log(predicts)+(1-targets)*np.log(1-predicts))
    @staticmethod
    def cross_error(targets, predicts):
        return -np.nan_to_num((targets/predicts)-(1-targets)/(1-predicts))
    @staticmethod    
    def delta(targets, predicts): 
        return (targets - predicts)


 




class ModelSelection(object):
    def __init__(self,Data,K):
        K_fold_Data = CrossValidationData.get_kfold_data(Data)
        return K_fold_Data



class CrossValidationData(object):
    @staticmethod
    def get_kfold_data(data, k):
        k_fold = [data[i:i+k] for i in range(0, len(data), k)]
        print 'k_fold', k_fold
        a = [(k_fold[item], np.delete(copy.copy(k_fold), item)) for item in xrange(len(k_fold))]
        print a        
        for item in xrange(len(k_fold)):
            valid = k_fold[item]            
            train = np.array(copy.copy(k_fold))
            train  = np.delete(copy.copy(k_fold), item)
            # del train[item]
            #print 'valid', valid
            #print '================'
            #print 'train', train            
        return k_fold
        



class PredictModel(object):
    def __init__(self):
        files = open(cld.fileDirectory.find_file_path() + 'Netparams', 'r')
        data = json.load(files)
        # cost = getattr(sys.modules[__name__], data["cost"])
        # net = Network(data["size"], cost=cost)
        #net.weights = [np.array(w) for w in data["weights"]]
        #net.biases = [np.array(b) for b in data["biases"]]
        return data
    
    '''
    def load(self, fileName):
        files = open(fileName, 'r')
        data = json.load(files)
        cost = getattr(sys.modules[__name__], data["cost"])
        net = Network(data["size"], cost=cost)
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
     '''
'''
    unzip = cld.unzipFiles()
    zipfile = unzip.unzip_files('mnist.pkl.gz')
    train_set, valid_set, test_set = zipfile['trainSet'], zipfile['validSet'],zipfile['testSet']
    # just testing my code,    
    trainSet, validSet, testSet = train_set, valid_set, test_set
    
    #print len(trainSet[0][0]),
    layers = [len(trainSet[0][0]), 15, 10]
    
    L = 15
    NetImageData = { 'train_x': trainSet[0][0:L],
                     'train_y': trainSet[1][0:L],       
                     'test_x': trainSet[0][0:L],
                     'test_y': trainSet[1][0:L],
                     'valid_x': validSet[0][0:L], 
                     'valid_y': validSet[1][0:L],
                     'layers': layers,
                     'epochs': 2,
                     'miniBatchSize': 1,        
                     'leraningRate': 0.001
                   }  
    #nn = ann.training(NetImageData)
'''                    
               


   