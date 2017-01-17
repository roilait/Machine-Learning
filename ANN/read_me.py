# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 22:47:40 2016

@author: moussa
"""

# -*- coding: utf-8 -*-
# the shape of the input of the function forward_prop should be (n,)
# the shape of the input of the function backward_prop should be (m,1)
"""
1 - pour un apprentissage non superviser il faut penser à la distribution de probabilite p(x),
2 - Donnee categorique: ex A = {femme, homme}, ou B = {oui, non, peut-etre},
converti donnee categorique vers donnee numerique (pretraitement): A = {1,0} ou {0, 1} et ,
B = {1,0,0} ou {0,1,0} ou {0,0,1},
3 - valeur manquante (il est possible qu'une valeur manque), 
       X = {1,2, ?, 5}=>pretraitement X = {1,2, '0,1', 5},
    si X = {1,2, 3, 5}=>pretraitement X = {1,2, '3,0', 5},
4 - Il est preferable de normaliser les donnee reelle pour le traitement, afin qu'elle prennemt de valeur 
    proche de 0,
    valeur trop eleve pourraient creer des instabilite numerique
5 - comment determiner si un Algorithm C est meilleur qu'un algorithm D ?
     - on peut regarder l'erreur de test, i.e, mesure de leur performence generaliser
     - Intervlle de confiance, on pourrait etre capable de dire que C est meilleur ou performant que D
     cours apprentissage automatique, 1.10 et 1.11
"""
# import matplotlib.pyplot as plt
import numpy as np
import json

class Training(object):
    def __init__(self, NetData, vectorize = True):
        train_x, train_y = NetData['train_x'], NetData['train_y']
        test_x, test_y = NetData['test_x'], NetData['test_y']
        valid_x, valid_y = NetData['valid_x'], NetData['valid_y']
        self.epochs, self.mbs = NetData['epochs'], NetData['miniBatchSize']        
        self.eta, self.alpha = NetData['eta'], NetData['alpha']
        layers, nbrOut = NetData['layers'], NetData['layers'][-1]         
               
        if vectorize:
            train_y = [ActFunction.vectorized(i,nbrOut) for i in train_y]
            if test_x!=[]: test_y = [ActFunction.vectorized(j, nbrOut) for j in test_y]
            if valid_x!=[]: valid_y = [ActFunction.vectorized(k, nbrOut) for k in valid_y]
        
        self.Train_data = ActFunction.change_data(train_x, train_y)
        if test_x!=[]: self.Test_data = ActFunction.change_data(test_x, test_y)
        if valid_x!=[]: self.Valid_data = ActFunction.change_data(valid_x, valid_y)
        
        # -----------------------
        # call the train function
        # -----------------------
        self.train_network(layers)

    def train_network(self, layers):
        Ann = NeuralNetwork(layers)
        """ Train the neural network using mini-batch stochastic gradient descent.  
        The 'training_data' is a list of tuples '(x, y)' representing the training 
        inputs and the desired outputs. If 'test_data' is provided then the network
        will be evaluated against the test data after each epoch, and partial 
        progress printed out. This is useful for tracking progress, but slows 
        things down substantially. """
        # if self.test_x:
        n = len(self.Train_data)        
        mbs = self.mbs
        # mbs, train_x, train_y = self.mbs, self.train_x, self.train_y
        epoch = 0 # epoch, iters, keep_looping = 0, -1, True
        training_cost, test_cost = [], []  
        test_accuracy = [] 
        while epoch < self.epochs:
            subData = [self.Train_data[i:i + mbs] for i in xrange(0, n, mbs)]
            # subData_y = [train_y[j:j + mbs] for j in xrange(0, n, mbs)]
            tr_avg_cost = []
            for data_x_y in subData:
                # the update function is called
                cost = Ann.update_params(data_x_y, self.eta, self.alpha)                
                tr_avg_cost.append(cost)                
            training_cost.append(np.mean(tr_avg_cost))             
            if self.Test_data:
                accuracy, cost = Ann.net_evalution(self.Test_data)
                test_cost.append(cost)
                test_accuracy.append(accuracy)
                m = len(self.Test_data)
                print "Epoch {0}: {1} / {2}".format(epoch, accuracy, m)
            # else:
                # print "Epoch {0} complete".format(epoch)            
            epoch+=1
        self.cost = {'train': training_cost, 'test': test_cost}
        self.acuracy = {'train':test_accuracy}
        
                
    def get_train_cost(self):
        return self.cost, self.acuracy
    
    def vectorized(self, j):
        """Return a 10-dimensional unit vector with a 1.0 in the jth position and 
        zeroes elsewhere.  This is used to convert a digit (0...9) into a 
        corresponding desired output from the neural network."""
        binary_output = np.zeros((1, self.nbrOutputNodes)) # np.zeros((10,1))
        binary_output[0][j] = 1.0
        
        return binary_output
        
class NeuralNetwork(object):    
    def __init__(self, layers):
        ''' The list layers contains the number of neurons in the respective layers of 
        the network.  For example, if the list was [3, 4, 5, 2] then it would be a 
        four-layer network, with the first layer containing 3 neurons, the second 
        layer 4 neurons, the third 5 layer and the third layer 1 neuron '''
        self.layers, self.num_layers = layers, len(layers), 
        self.biases = [np.random.randn(k,1) for k in layers[1:]]
        self.weights = [np.random.rand(x,k)/np.sqrt(x) for x,k in zip(layers[:-1],layers[1:])]
        self.matrix = {'forward': [], 'backward': []}
        
    # forward propagation function    
    def forward_prop(self, x):        
        activation = np.array([x])  
        # list to store all the z vectors, layer by layer
        # self.noActiv = []   
        # list to store all the activations, layer by layer
        self.noActiv, self.matrix['forward'] = list(), list()
        self.matrix['forward'].append(activation)
        for w, b in zip(self.weights, self.biases):
            f = np.dot(activation, w) + b.T
            self.noActiv.append(f)
            activation = ActFunction.sigmoid(f)
            self.matrix['forward'].append(activation)
        
        return self.matrix['forward'][-1]    
   
    # backward propagation function   
    def back_propa(self, inputs, truthVal):
        # the forward propagation function is called
        self.forward_prop(inputs)
        """ Return a tuple '(nabla_b, nabla_w)' representing the gradient for the cost 
        function C_x.  'delta_b' and 'delta_w' are layer-by-layer lists of numpy arrays, 
        similar to 'self.biases' and 'self.weights'. """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        Inputs, targets, predicts = self.noActiv[-1], truthVal, self.matrix['forward'][-1]        
        cost = QuadratiCost.mse(targets, predicts)        
        delta = QuadratiCost.delta(Inputs, targets, predicts)
        #cost = CrossEntropyCost.cross_cost(targets, predicts)
        #delta = CrossEntropyCost.delta(targets, predicts).T
        nabla_b[-1] = delta
        nabla_w[-1] = (np.dot(delta, self.matrix['forward'][-2])).T
        
        for i in xrange(2, self.num_layers):
            k = self.noActiv[-i]            
            back = ActFunction.sigmoid_prime(k).T  
            delta = np.dot(self.weights[-i+1], delta)*back
            nabla_b[-i] = delta
            nabla_w[-i] = (np.dot(delta, self.matrix['forward'][-i-1])).T 
            
        return cost, nabla_w, nabla_b  
        
    # update parameters function
    def update_params(self, subdata, eta, alpha): 
        """ Update the weights and biases of the network by applying
        gradient descent using backpropagation to a tran_x and train_y.
        'alpha' is the learning rate. """
        avg_cost = []
        for x, y in subdata:
            # x.shape should be (n,)
            # y.transpose().shape should be (1,m), [[y1 y2 y3 ...yn]]           
            # the backward propagation function is called
            cost, delta_nabla_w, delta_nabla_b = self.back_propa(x, y) 
            self.weights = [w + alpha*w + eta*nw for w, nw in zip(self.weights, delta_nabla_w)] 
            self.biases = [b + alpha*b + eta*nb for b, nb in zip(self.biases, delta_nabla_b)]
            
            avg_cost.append(cost) 
            # print 'y :',y, 'yhat :',self.matrix['forward'][-1]
        
        return np.mean(avg_cost)
        
    def net_evalution(self, test_data):
        #cost = QuadratiCost.mse(targets, predicts)        
        #cost = CrossEntropyCost.cross_cost(targets, predicts)
        results = [(np.argmax(self.forward_prop(x)), np.argmax(y)) for (x,y) in test_data]
        test_cost = np.mean([np.mean(QuadratiCost.mse(y, self.forward_prop(x))) for x,y in test_data])
        
        return sum(int(x==y) for x, y in results)/float(len(test_data)), test_cost
        
    def accuracy(self, truthVal):
        er = 0.04 # acceptable error
        y = truthVal[0]
        y_pred = self.matrix['forward'][-1][0]
        result = [1 for i in range(len(y)) if abs(y[i]-abs(y_pred[i]))<=er]
        accur = sum(result)/len(y)
        
        return accur
        
    # def evaluate(self, test_data, error=True):
        """Return the number of test inputs for which the neural network outputs the 
        correct result. Note that the neural network's output is assumed to be the 
        index of whichever neuron in the final layer has the highest activation."""
        #test_results = [(np.argmax(forward_prop(x, rturn=True)), y) for (x, y) in test_data]:
                                           
 
       #return sum(int(x==y) for (x,y) in test_results)
 
# this class return a instance of network 
class LoadingNetwork (object):
    def saveNetwork(self, fileName, epoch):
        data = {'size': self.size,
                'epoch':epoch,
                'weights':[w.tollist() for w in self.weights],
                'biases':[b.tolist() for b in self.biases],
                'cost': str(self.cost.__name__)
                }
                
        files = open(fileName, "w")
        json.dump(data, files)
        files.close()
class QuadratiCost(object):
    @staticmethod
    def mse(targets, predicts): # evaluate quadratic cost 
        return np.mean(0.5*(targets-predicts)**2)
    @staticmethod
    def mse_deriv(targets, predicts):
        return targets-predicts # delta error
    @staticmethod
    def delta(inputs, targets, predicts):
        z_prime = ActFunction.sigmoid_prime(inputs).T
        return z_prime*(targets-predicts).T
        
class CrossEntropyCost(object):
    # the np.nan_to_num function ensures that np.log(0) evaluates to o instead of nan
    @staticmethod    
    def cross_cost(targets, predicts):
        return -np.nan_to_num(targets*np.log(predicts)+(1-targets)*np.log(1-predicts))
    @staticmethod
    def cross_error(targets, predicts):
        return -np.nan_to_num((targets/predicts)-(1-targets)/(1-predicts))
    @staticmethod    
    def delta(targets, predicts):
        return (targets - predicts)
 
class ActFunction(object):
    # sigmoid function
    @staticmethod   
    def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))        
    # sigmoid prime function    
    @staticmethod
    def sigmoid_prime(pr):
        return ActFunction.sigmoid(pr)*(1 - ActFunction.sigmoid(pr))
    @staticmethod    
    def vectorized(j, nbr):
        binary_output = np.zeros((1, nbr))
        binary_output[0][j] = 1.0
        return binary_output
    @staticmethod
    def change_data(inputs, outputs):
        return [(inputs[i], outputs[i]) for i in xrange(len(inputs))]
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
               


   