#!/usr/bin/python

import theano,os,csv,random
import theano.tensor as T
import numpy as np
from itertools import izip
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# classes imported 
import cleanData as cld
import NeuralNetwork as ann

# -------------------------
# -------- Class ----------
# -------------------------

class Functions(object):
    # ------------------------
    def __init__(self, classifier):
        self.classifier = classifier
        
    # ---------------------------------------------------    
    def errors(self, truthVal, predVal, getAccuracy = True): # y is the desired Output
        """: type y: theano.tensor.TensorType
        : param y: corresponds to a vector that gives for each example 
         the correct label"""
        # check if y has same dimension of y_pred
        if (len(truthVal)!= len(predVal)):
            raise TypeError(
                'y should have the same shape as self.y_pred',
                  ('y', truthVal.type, 'predVal', predVal.type))
        # check if y is of the correct datatype
        else:
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            #print 'truth_val, pred_val', truth_val, pred_val
            errors = sum([abs(target-output) for target, output 
                          in zip(truthVal, predVal)])
            #print '====', target, '-', output            
            print 'error', errors
            error = T.mean(T.neq(predVal, truthVal))
            Accurancy = error.eval() # if we want to have just the error
            if (getAccuracy):
                Accurancy = T.mean(T.eq(predVal, truthVal))
                Accurancy = Accurancy.eval()
                # print '=====', er￼ror.eval()
            return Accurancy
            
    def mnistToDigital(self, mnistOut):
        out = [0]*10
        out[mnistOut] = 1
        return out
    # --------------------------------------------   
    def update_params(self, parameters, gradients, learningRate):
        # compute list of fine-tuning updates
        updates = [
                     (param, param - gparam*learningRate)
                      for param, gparam in zip(parameters, gradients)
                   ]
        return updates
        
    # ------------------------------   
    def cost_function(self, X, truthVal, params):
        weight, bias = params[0], params[1]
        # Define mathematical expression:
        z = T.dot(X, weight) + bias       
        predVal = z        
        if (self.classifier=='classification'):
            predVal = T.nnet.sigmoid(z)
            # predVal = T.tanh(z)
            # predVal = T.nnet.softmax(z)          
        # cross-entropy cost function            
        cost = T.sum(-(truthVal*T.log(predVal) + (1-truthVal)*T.log(1-predVal)))
        # Mean Squared Error = Quadratic cost        
        # cost = T.sum((y-self.output_pred)**2) # The cost to minimize
        
        return cost, predVal 
        
        
# ------------------------------
# ------------ Class -----------
# ------------------------------
        
class Regression(object):
    def __init__(self, cleaneData, features):
        # -------------- Classes --------------
        self.CallFunct = Functions(features["classifier"])
        # ------------------------------------- 
        self.features = features
        learn_rate = features['learningRate']
        # self.Data = np.array(cleaneData['train_set'])                
        self.trainSet, n_in =  cleaneData['trainSet'], len(cleaneData['trainSet'][0])
        self.trainVal, n_out = cleaneData['trainVal'], len(cleaneData['trainVal'][0])
        
        self.validSet, self.validVal = cleaneData['validSet'], cleaneData['validVal'] 
        weight = theano.shared(
                  value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                  name = 'Input Weight', borrow = True)                              
        # initialize the biases b as a vector of n_out 0s
        bias = theano.shared(
                value = np.zeros((n_out), dtype=theano.config.floatX),
                name = 'Input bias', borrow = True) 
        self.params = [weight, bias] # that should be updated 
       
        #Define theano variables:
        x, truthVal = T.matrix('x'), T.matrix('truth val') # the dim ou the input is N x n_in
        #truth_val = T.matrix('truth_val')                   # the dim of the desired output is n_out 
        
        train_cost, pred_val = self.CallFunct.cost_function(x, truthVal, self.params) # training cost
        #errorAccuracy = 0 #CallFunct.errors(truth_val, self.params) # training cost
        
        # print y_hat   
        # theano.printing.pp(y_hat)              
        grads = T.grad(train_cost, self.params)        
        self.train_model = theano.function( 
                                  inputs = [x, truthVal],
                                  outputs = [pred_val, train_cost],
                                  u￼pdates = self.CallFunct.update_params(self.params, grads, learn_rate)
                             ) 
        
        self.test_model = theano.function( inputs = [x, truthVal],
                                           outputs = [pred_val, train_cost]
                                          )                      
        self.validate_model = theano.function( inputs = [x, truthVal],
                                               outputs = [pred_val, train_cost]
                                              )                                              
        #Iterate through all inputs and find outputs:
        
    '''   
    def prediction(self,data_x, data_y):
        valid_cost, pred_y = self.cost_function(data_x, data_y) # training cost
        return valid_cost, pred_y
        
    def validation(self,val_x, val_y):
        # execute a prediction and check how many result are OK
        valid_cost, pred_y = self.prediction(val_x, val_y)
        nr_good = 0
        for i in xrange(len(val_y)):
            
            if (pred_y[i]==val_y[i]):
                nr_good += 1
        return valid_cost, pred_y, nr_good

    ''' 
    def train(self):
        train_set, train_val = self.trainSet, self.trainVal
        valid_set, valid_val = self.validSet, self.validVal
        
        n_epochs = self.features["n_epochs"]
        #self.regression(features)        
        epoch, iters, keep_looping = 0, -1, True
        tr_avg_cost, val_avg_cost = list(), list() 
        tr_avg_accuracy, val_avg_accuracy = list(), list()
        while (epoch < n_epochs) and (keep_looping):
            epoch, iters = epoch + 1, iters + 1
            tr_cost, tr_accuracy, val_cost, val_accuracy = list(), list(), list(), list()
            
            for iteration in range(0, len(train_set)):
                Inputs = np.array([train_set[iteration]])               
                Outputs = np.array([train_val[iteration]])
                # print 'Outputs', Outputs
                predVal, cost_iter = self.train_model(Inputs, Outputs)
                #self.CallFunct.errors(outputs[0], predVal[0])
                #accuracy.append(self.CallFunct.errors(Outputs[0], pred_val[0]))
                #print self.CallFunct.errors(outputs[0], predVal[0])
                print iteration
                print 'test', Outputs, '--', predVal
                #print '=========================================================================='
            
                tr_cost.append(cost_iter)
              
            for iterations in xrange(len(valid_set)):
                Inputs = np.array([valid_set[iterations]])               
                Outputs = np.array([valid_val[iterations]])
                predValu, cost_viter = self.validate_model(Inputs, Outputs)
                #print Outputs, '-', predValu, '-', epoch 
                val_cost.append(cost_viter)
                  
            tr_avg_cost.append(np.mean(sum(tr_cost)))
            val_avg_cost.append(np.mean(sum(val_cost)))
            #avg_accuracy.append(np.mean(sum(accuracy)))
            #if (iters==len(self.Data[2])):
                #iters = 0
            #val_x, val_y = np.array([self.Data[2][iters]]), np.array([self.Data[3][iters]])

            #valcost, y_hat, nr_good = self.validation(val_x, val_y)
            #print valcost.get_value()[0], 
            #val_cost.append(valcost)
            #good.append(nr_good)
            #print epoch, avg_cost,￼ precision
            # show the regression figure
            #fig = show_figure(self.params, self.Data)
            #fig.plot_3D(bias, weight, Cost)
            #print self.Data
            #fig.plot_2D()
        plt.figure(1)
        plt.plot(tr_avg_cost)
        plt.plot(val_avg_cost)
        #plt.figure(2)
        #plt.plot(good) 

        #fig.plot_3D(bias, weight, Cost)

# ----------------------
# ------ Class ---------
# ----------------------

if __name__=="__main__":
    
    features = { 
        "fileName": 'iris', # the name of the file,
        "fileExtension": '.data', # the extension of the file,        
        "lenght": 4, # dim of the input vector,
        "learningRate": 0.00001,# learning rate,
        "n_epochs": 150, # the number of the epochs,
        "classifier": 'classification' # choice regression or classification
    }
    #dcl = cld.clean_data()
    #pourcentage = 0.7
    #cleaneData = dcl.get_cleaned_data(features, pourcentage)
    
    # print cleaneData
    #reg = Regression(cleaneData, features)
    #reg.train()
    

    
    '''
    import matplotlib.cm as cm
    
    train_x, train_y = train_set
    print train_set[0]
    print train_set[1]
    plt.imshow(train_x[0].reshape((28, 28)), cmap = cm.Greys_r)
    plt.show()
    '''
    #layers = [2,3,4, 2]
    #nn = ann.neuralNetwork(layers)
    #a = np.array([1,2])
    #print 'a', a.shape
    #nn.forward_propagation(a)
    ''''
    Ld = cld.clean_data()
    Data = Ld.get_features(features)
    
    print Data['train_set'], '-', len(Data['train_set']) 
    print "================"
    print Data['train_val'], '-', len(Data['train_val'])
    print "+++++++++++++"
    print Data['test_set'], '-', len(Data['test_set'])
    print "================"
    print Data['test_val'], '-', len(Data['test_val'])
    
    print Data['realLabels']           
    '''    
    #load_data = load_data()
    '''
    we should fix the lenght to separe the inputs and outputs data
     from the file'''
     #Data_set = load_data.get_data_set(features)
    #print 'Data_set',Data_set
    
    
    # tstart = LogisticRegression(features)
    
    # tstart.train(features)
            
        