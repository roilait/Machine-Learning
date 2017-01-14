#!/usr/bin/python
# -*- coding: utf-8 -*-
# classes imported 
import matplotlib.pyplot as plt
import getData as getd

# IMPORT CLASSES
import NeuralNetwork as ann
import annAlgo as cls
import Regression as reg
import numpy as np
# Create random data with numpy

if __name__=="__main__":  
    # SET THE PARAMETERS 
    feats = { 'TrainEpochs':300,
              'TrainBatchSize':20, 
              'TestBatchSize':5,
              'displayStep':1, 
              'learning_rate':0.001,
              'k_fold':5
             }
    n_classes = 10
    Data = getd.CvsFileMnist.manupulations(n_classes)
    # -----------------------------------------------------------
    # ----------------------- mode training ---------------------
    # -----------------------------------------------------------    
    feats['hidden_layers'] = [256] # set the layers of the neural
    hidden_layers = [256, 256]
    
    print Data['train_x']

    
    
    # cls.Cls_model(Data, feats, hidden_layers)
    # cls.NeuralNetworkModel(Data)


    # CROSS VALIDATION METHOD
    #cv.CrossValidation(Data, feats)
    
    # algo.CrossValidation(NetImageData_3, k_fold, feats)   