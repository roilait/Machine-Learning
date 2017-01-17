#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# THE CLASSES IMPORTED
import getData as getd

logs_path = getd.Path.find_file_path()

class NeuralNetworkModel(object):
    def __init__(self, Data, HLayers):
        global train_x, train_y, test_x, test_y, valid_x, valid_y 
        global displayStep, batchSize, train_epochs, regression, learning_rate
        
        batchSize = 10
        train_epochs = 10
        regression = True
        displayStep = 5 
        learning_rate = 0.01
        
        train_x, train_y = Data['train_x'], Data['train_y']
        test_x, test_y = Data['test_x'], Data['test_y']
        valid_x, valid_y = Data['valid_x'], Data['valid_y']
        
        activation_function = {'actFctionHL':tf.nn.sigmoid, 'actFctionOut': tf.nn.softmax}
        if regression:
            activation_function = {'actFctionHL':tf.nn.relu, 'actFctionOut': None}
            # MAKE UP SOME REAL DATA
            train_x = np.linspace(-1, 1, 300)[:,np.newaxis]
            noise = np.random.normal(0, 0.05, train_x.shape)
            train_y = np.square(train_x)-0.5 + noise 
            HLayers = [10]
        
        n_inputs, n_classes = len(train_x[0]), len(train_y[0]) 
        # PREPARING THE LAYERS LIST OF THE NETWORK 
        layers = []
        layers.append(n_inputs)
        layers.extend(HLayers)
        layers.append(n_classes)
        # NEURAL NETWORK MODEL CALLED
        NeuralNetworkModel.neural_network_model(layers, activation_function)
        
    @staticmethod
    def neural_network_model(layers, activation_function, Lys = []):
        global inputs, outputs, loss, predict, accuracy, train_step, merged           
        # DEFINE THE INPUTS AND OUTPUTS PLACEHOLDER FOR THE NETWORK AND TENSORBOARD
        with tf.name_scope('inputs'):
            inputs = tf.placeholder(tf.float32, [None, layers[0]], name = 'x_inputs')
            outputs = tf.placeholder(tf.float32,[None, layers[-1]], name = 'y_output')
        # INITIALIZATION OF THE WEIGHTS AND BIASES
        with tf.name_scope('Hyper_params'):
            with tf.name_scope('weights'):
                weights = ModelClasses.get_weights(layers)                
            with tf.name_scope('biases'):
                biases = ModelClasses.get_biases(layers)
        # FORWARD PROPOAGATION, BUILD THE PREDICTION MODEL 
        parms = {'weights': weights, 'biases': biases}      
        with tf.name_scope('Forward_propagation'):
            predict = ModelClasses.predict_model(inputs, parms, activation_function)
            # CREATE A HISTOGRAM SUMMARY TO VIZUALIZE THE OUTPUT LAYERS
            tf.histogram_summary('Output_layer', predict)
        # COMPUTE THE COST FUNCTION
        with tf.name_scope('Loss'):
            loss = ModelClasses.cost_function(outputs, predict)
            loss = tf.reduce_mean(loss)        
        # COMPUTE THE GRADIENTS
        with tf.name_scope('Train_step'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_step = optimizer.minimize(loss)
        # COMPUTE THE ACCURACY
        with tf.name_scope('Accuracy'):
            with tf.name_scope('is_correct'):
                is_correct = tf.equal(tf.argmax(outputs, 1),tf.argmax(predict, 1))
            with tf.name_scope('Accuracy'):
                accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        # CREATE THE SUMMARY FOR OUR COST AND ACCURACY
        tf.scalar_summary('Loss', loss)       
        tf.scalar_summary('Accuracy', accuracy)   
        
        # MERGE ALL THE SUMMARIES AND WRITE THEM OUT IN logs FILE
        merged = tf.merge_all_summaries()
        # ALSO, WE TO INITIALIZE ALL VARIABLES, IT IS A VERY IMPORTANT        
        init_variables = tf.initialize_all_variables()
         
        # THIS FUNCTION IS CALLED IF WE ARE USING REGRESSION
        if regression:
            RegressionModel.regression(init_variables)
        else:
            print  ClassificationModel.classification(init_variables)


class RegressionModel(object):
    # -------------------------
    # -- REGRESSION FUNCTION --
    # -------------------------
    @staticmethod
    def regression(init):
        # PLOT THE DATA              
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(train_x, train_y)
        plt.xlim(-1.5,1.5)
        plt.ylim(-0.7, 0.7)
        plt.ion()
        plt.show()
        with tf.Session() as sess:
            sess.run(init)
            # DELET THE logs FILE, BEFORE TO ADD THE GRAPH
            getd.DeleteFileOrFolder(logs_path+'logs/')
            train_writer = tf.train.SummaryWriter(logs_path+'logs/train', sess.graph)
            for step in xrange(1000):
                sess.run(train_step, feed_dict={inputs:train_x,outputs:train_y})
                if step%displayStep==0:
                    train_result = sess.run(merged, feed_dict={inputs:train_x, outputs:train_y})
                    train_writer.add_summary(train_result, step)
                    # TO VISUALIZE THE RESULT AND IMPROVEMENT                    
                    try:
                        ax.lines.remove(lines[0])
                    except Exception:
                        pass
                    #cost = sess.run(loss, feed_dict = {xs: x_data, ys: y_data})
                    predict_value = sess.run(predict, feed_dict={inputs:train_x})
                    # PLOT THE PREDICTION
                    lines = ax.plot(train_x, predict_value, 'r-', lw = 3) 
                    plt.pause(0.1)
                    
                   
class ClassificationModel(object):
    @staticmethod
    def classification(init):
        with tf.Session() as sess:
            # CREATE INITIALIZED VARIABLE
            sess.run(init) 
            # DELET THE logs FILE, BEFORE TO ADD THE GRAPH
            getd.DeleteFileOrFolder(logs_path+'logs/')
            # SUMMARY WRITER GOES HERE
            train_writer = tf.train.SummaryWriter(logs_path+'logs/train', sess.graph)
            test_writer = tf.train.SummaryWriter(logs_path+'logs/test') 
            # PARAMETERS INITIALIZATION                       
            epoch = 0
            while epoch < train_epochs:
                # LOOP OVER ALL BATCHES
                iters = 0
                while iters<len(train_x):
                    start, end = iters, iters + batchSize
                    batch_x, batch_y = train_x[start:end], train_y[start:end]
                    _, cost = sess.run([train_step, loss], feed_dict={inputs:batch_x, outputs:batch_y})             
                        
                    iters += batchSize
                
                train_result = sess.run(merged, feed_dict={inputs:train_x, outputs:train_y})
                train_writer.add_summary(train_result, epoch)
                
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.3f}".format(cost))
                
                epoch += 1     
        
        
class ModelClasses(object):       
    # WEIGHT FUNCTION AND INITIALIZATION
    @staticmethod 
    def get_weights(layers):
        # INITIALIZE THE WEIGHTS WITH APPROPRIATE INITIALIZATION
        def init_weights(shape):
            return tf.Variable(tf.random_normal(shape), name='weighs')
        weights = {'w%s'%i: init_weights([layers[i-1], layers[i]]) for i in xrange(1, len(layers))} 
            
        return  weights      
    # BIASES FUNCTION AND INITIALIZATION
    @staticmethod 
    def get_biases(layers):
        # INITIALIZE THE BIASES WITH APPROPRIATE INITIALIZATION
        def init_biases(shape):
            return tf.Variable(tf.random_normal(shape), name='biases')
        biases = {'b%s'%i: init_biases([layers[i]]) for i in xrange(1, len(layers))}
        
        return biases    
    # COMPUT THE COST FUNCTION
    @staticmethod
    def cost_function(outputs, ys_pred):
        if regression:
            # SQUARED LOSS FUNCTION
            loss = tf.reduce_sum(tf.square(outputs-ys_pred), reduction_indices=[1])
        else:
            # SOFTMAX CROSS ENTROPY (COST FUNCTION)
            #loss = -tf.reduce_sum(ys*tf.log(tf.clip_by_value(ys_pred, 1e-10, 1.0)))
            loss = -tf.reduce_sum(outputs* tf.log(ys_pred), reduction_indices=[1])
            # loss = tf.nn.softmax_cross_entropy_with_logits(ys_pred, ys)
            # loss = -tf.reduce_sum(ys*tf.log(ys_pred), reduction_indices=[1])
            # loss = tf.nn.softmax_cross_entropy_with_logits(pred_outputs, outputs)          
            
        return loss    
    
    # CREATE MODEL
    @staticmethod
    def predict_model(inputs, parms, activation_function):        
        I, Ls = 1, {'layer%s'%ls:'' for ls in xrange(1, len(parms['weights'])+1)}
        Wgs, bias = parms['weights'], parms['biases']
        act_fction_HLayers = activation_function['actFctionHL']
        act_fct_out = activation_function['actFctionOut']         
        while I < len(parms['weights']): 
            # HISTOGRAM OF THE WEIGHTS AND BIASES
            #variable_summaries(W['w%s'%(I)])
            #variable_summaries(bias['b%s'%(I)])
            #tf.histogram_summary('layer%s '%(I)+'/Weights%s'%(I), W['w%s'%(I)])
            #tf.histogram_summary('layer%s '%(I)+'/biases%s'%(I), bias['b%s'%(I)])
            if (I==1):
                Ls['layer%s'%(I)] = tf.add(tf.matmul(inputs, Wgs['w%s'%(I)]), bias['b%s'%(I)])
                if act_fction_HLayers is not None:
                    Ls['layer%s'%(I)] = act_fction_HLayers(Ls['layer%s'%(I)])
                # CREATE A SAMMARY TO VISUALIZE THE FIRST LAYER RELU ACTIVATION
                tf.histogram_summary("activation_layer %s"%(I), Ls['layer%s'%(I)])
            else:
                Ls['layer%s'%(I)] = tf.add(tf.matmul(Ls['layer%s'%(I-1)], Wgs['w%s'%(I)]), bias['b%s'%(I)])
                if act_fction_HLayers is not None:
                    Ls['layer%s'%(I)] = act_fction_HLayers(Ls['layer%s'%(I)])
                # CREATE A SAMMARY TO VISUALIZE THE FIRST LAYER RELU ACTIVATION
                tf.histogram_summary("activation_layer%s "%(I), Ls['layer%s'%(I)])
            I += 1   
             
        # COMPUTING THE OUTPUT LAYER VALUE  
        prediction = tf.add(tf.matmul(Ls['layer%s'%(I-1)], Wgs['w%s'%(I)]), bias['b%s'%(I)])
        # CREATE A SAMMARY TO VISUALIZE THE FIRST LAYER RELU ACTIVATION
        # variable_summaries(W['w%s'%(I)])
        # variable_summaries(bias['b%s'%(I)])
        # tf.histogram_summary('layer%s '%I+'/Weights%s'%(I), W['w%s'%I])
        # tf.histogram_summary('layer%s '%I+'/biases%s'%(I), bias['b%s'%I])
        if act_fct_out is not None:
            prediction = act_fct_out(prediction)
                            
        return prediction
    
    
    @staticmethod    
    def compute_accuracy(v_xs,v_ys, sess):
        # % OF CORRECT ANSWERS FOUND IN BATCH
        y_pred = sess.run(predict, feed_dict = {xs: v_xs})
        is_correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        result = sess.run(accuracy,feed_dict = {xs: v_xs, ys: v_ys})
        
        return result
