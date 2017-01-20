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
import tensorflow as tf
# Create random data with numpy

if __name__=="__main__":  
    # SET THE PARAMETERS 
    dictInput = {'batchSize': 10,
                 'trainEpochs': 4,
                 'regression': False,
                 'displayStep': 1,
                 'learningRate': 0.01,
                 'useDropout': True,
                 'keepProbValue':0.5
                }
    # NUMBER OF CLASSES
    n_classes = 10 
    Data = getd.CvsFileMnist.manupulations(n_classes)
    # -----------------------------------------------------------
    # ----------------------- mode training ---------------------
    # -----------------------------------------------------------    
    hidden_layers = [256, 256]

    # cls.Cls_model(Data, feats, hidden_layers)
    activation_function = {'actFctionHL':tf.nn.sigmoid, 'actFctionOut': tf.nn.softmax}
    cls.SaverNeuralNetworkModel(Data, hidden_layers, dictInput, activation_function)
 
