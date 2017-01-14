#!/usr/bin/python
# -*- coding: utf-8 -*-
# http://jorditorres.org/first-contact-with-tensorflow/
# https://www.analyticsvidhya.com/blog/2016/10/an-introduction-to-implementing-neural-networks-using-tensorflow/
# https://www.youtube.com/watch?v=vq2nnJ4g6N0
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_advanced.py
# http://camron.xyz/index.php/2016/10/03/nutshell-all-the-models/
# https://www.youtube.com/watch?v=8XKSQTCzXEI&list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f&index=11
# https://www.youtube.com/watch?v=G5RY_SUJih4

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# THE CLASSES IMPORTED
import getData as getd

logs_path = getd.Path.find_file_path()

class NeuralNetworkModel(object):
    def __init__(self, Data):
        global xs, ys, x_data, y_data, x_test, y_test, x_valid, y_valid 
        global displayStep, batchSize, train_epochs
        
        batchSize = 2
        train_epochs = 1000
        regression = False 
        displayStep = 5 
        HLayers = [8]
        
        x_data, y_data = Data['train_x'], Data['train_y']
        x_test, y_test = Data['test_x'], Data['test_y']
        x_valid, y_valid = Data['valid_x'], Data['valid_y']
        
        activation_function = {'act_fction':tf.nn.relu, 'OutActi_fction': tf.nn.softmax}
        if regression:
            activation_function = {'act_fction':tf.nn.relu, 'OutActi_fction': None}
            # MAKE UP SOME REAL DATA
            x_data = np.linspace(-1, 1, 300)[:,np.newaxis]
            noise = np.random.normal(0, 0.05, x_data.shape)
            y_data = np.square(x_data)-0.5 + noise       
        
        n_inputs, n_classes = len(x_data[0]), len(y_data[0]) 
        # PREPARING THE LAYERS LIST OF THE NETWORK 
        layers = []
        layers.append(n_inputs)
        layers.extend(HLayers)
        layers.append(n_classes)
        NeuralNetworkModel.neural_network_model(layers, activation_function, regression)
        
    @staticmethod
    def neural_network_model(layers, activation_function, regression, learning_rate = 0.1, Lys = []):
        global xs, ys, loss, predict, accuracy           
        # NETWORK PARAMETERS       
        # DEFINE THE INPUTS AND OUTPUTS PLACEHOLDER FOR THE NETWORK AND TENSORBOARD
        with tf.name_scope('inputs'):
            xs = tf.placeholder(tf.float32, shape = [None, layers[0]], name = 'x_inputs')
        with tf.name_scope('outputs'):
            ys = tf.placeholder(tf.float32, shape = [None, layers[-1]], name = 'y_output')
        # INITIALIZATION OF THE WEIGHTS AND BIASES
        with tf.name_scope('Hyper_params'):
            with tf.name_scope('weights'):
                weights = ModelClasses.get_weights(layers)
            with tf.name_scope('biases'):
                biases = ModelClasses.get_biases(layers)
        # FORWARD PROPOAGATION, BUILD THE PREDICTION MODEL 
        parms = {'weights':weights, 'biases':biases}
        # NET FORWARD PROPAGATION       
        with tf.name_scope('Feed_forward'):
            predict = ModelClasses.predict_model(xs, parms, activation_function)
            # CREATE A HISTOGRAM SUMMARY TO VIZUALIZE THE OUTPUT LAYERS
            tf.histogram_summary('Output_layer', predict)
        # COMPUTE THE COST FUNCTION
        with tf.name_scope('Loss'):
            loss = ModelClasses.cost_function(ys, predict, regression)
            loss = tf.reduce_mean(loss)
            tf.scalar_summary('Loss', loss)
        # COMPUTE THE ACCURACY OF THE MODEL
        with tf.name_scope('Accuracy'):
            accuracy = tf.equal(tf.argmax(ys,1),tf.argmax(predict,1))
            accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
            tf.scalar_summary('Accuracy', accuracy)
        # COMPUTE THE GRADIENTS
        with tf.name_scope('Train_step'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            # Op TO CALCULATE EVERY VARIABLE GRADIENT
            grads = tf.gradients(loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            # Op TO UPDATE ALL VARIABLES ACCORDING TO TEIR GRADIENT
            apply_update = optimizer.apply_gradients(grads_and_vars = grads)
        # TRAINABLES VARABLES
        for var in tf.trainable_variables():
            tf.histogram_summary(var.name, var)           
        # SUMMARIZE ALL GRADIENTS
        for grad, var in grads:
            tf.histogram_summary(var.name+'/gradient', grad)  
        # HERE, WE ARE GOING TO MERGE ALL SUMMARIES TO A SINGLE OP
        merged = tf.merge_all_summaries()
        
        # ALSO, WE TO INITIALIZE ALL VARIABLES, IT IS A VERY IMPORTANT
        init = tf.initialize_all_variables()
        
        # THIS FUNCTION IS CALLED IF WE ARE USING REGRESSION
        if regression:
            RegressionModel.regression(init, loss,apply_update, merged)
        else:
            ClassificationModel.classification(init, apply_update, merged)


class RegressionModel(object):
    @staticmethod
    def regression(init, loss, apply_update, merged):
        # PLOT THE DATA
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(x_data, y_data)
        plt.ion()
        plt.show()
        with tf.Session() as sess:
            sess.run(init)
            # DELET THE logs FILE, BEFORE TO ADD THE GRAPH
            getd.DeleteFileOrFolder(logs_path+'logs/')
            train_writer = tf.train.SummaryWriter(logs_path+'logs/train', sess.graph)            
            for step in xrange(1000):
                sess.run(apply_update, feed_dict = {xs: x_data, ys: y_data})
                if step%displayStep==0:
                    train_result = sess.run(merged, feed_dict = {xs: x_data, ys: y_data})
                    train_writer.add_summary(train_result, step)
                    # TO VISUALIZE THE RESULT AND IMPROVEMENT                    
                    try:
                        ax.lines.remove(lines[0])
                    except Exception:
                        pass
                    #cost = sess.run(loss, feed_dict = {xs: x_data, ys: y_data})
                    predict_value = sess.run(predict, feed_dict={xs: x_data})
                    # PLOT THE PREDICTION
                    lines = ax.plot(x_data, predict_value, 'r-', lw = 3) 
                    plt.pause(0.1)
                    
                    
class ClassificationModel(object):
    @staticmethod
    def classification(init, apply_update, merged):
        # NOW LET US CREATE A SESSION, AND RUN OUR NEURAL NETWORK SESSION
        batches_x, batches_y = ModelClasses.get_batches(x_data, y_data, batchSize)
        total_batch = len(batches_x)
        with tf.Session() as sess:
            # CREATE INITIALIZED VARIABLE
            sess.run(init) 
            # DELET THE logs FILE, BEFORE TO ADD THE GRAPH
            getd.DeleteFileOrFolder(logs_path+'logs/')
            # SUMMARY WRITER GOES HERE
            train_writer = tf.train.SummaryWriter(logs_path+'logs/train', sess.graph)
            test_writer = tf.train.SummaryWriter(logs_path+'logs/test', sess.graph) 
            # PARAMETERS INITIALIZATION                       
            train_cost, train_accuracy = [], [] 
            test_cost, test_accuracy = [], [] 
            epoch = 0
            while epoch < train_epochs:
                avg_cost, avg_acc = 0., 0.
                # LOOP OVER ALL BATCHES
                for i, batch_x, batch_y in zip(range(total_batch), batches_x, batches_y):
                    dct = {xs: x_data, ys: y_data}
                    _, train_result, cost = sess.run([apply_update, merged, loss], feed_dict = dct)
                    # WRITE LOGS AT EVERY ITERATION
                    train_writer.add_summary(train_result, epoch)    
                    avg_cost += cost/total_batch 
                train_cost.append(avg_cost)
                
                if x_test is not None:
                    if(epoch + 1) % displayStep==0:                        
                        test_result = sess.run(merged, feed_dict = {xs: x_test, ys: y_test})                    
                        test_writer.add_summary(test_result, epoch)
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.3f}".format(avg_cost))
                        
                epoch += 1
                          
        # return train_cost, test_cost, train_accu, test_accu        
        
        
class ModelClasses(object): 
    # MAKE UP THE BATCHES 
    @staticmethod
    def get_batches(data_x, data_y, batch_size):
        batches_x = [data_x[i:i + batch_size] for i in xrange(0, len(data_x), batch_size)]
        batches_y = [data_y[j:j + batch_size] for j in xrange(0, len(data_x), batch_size)]
        
        return batches_x, batches_y        
    # WEIGHT FUNCTION AND INITIALIZATION
    @staticmethod 
    def get_weights(layers):
        def init_weight(shape):
            return tf.Variable(tf.random_normal(shape), name='weighs')
        weights = {'w%s'%i:init_weight([layers[i-1], layers[i]]) for i in xrange(1, len(layers))}             
        return  weights      
    # BIASES FUNCTION AND INITIALIZATION
    @staticmethod 
    def get_biases(layers):
        def init_biases(shape):
            return tf.Variable(tf.random_normal(shape), name='biases')
        biases = {'b%s'%i:init_biases([layers[i]]) for i in xrange(1, len(layers))}
        return biases    
    # COMPUT THE COST FUNCTION
    @staticmethod
    def cost_function(ys,ys_pred, regression):
        if regression:
            # SQUARED LOSS FUNCTION
            loss = tf.reduce_sum(tf.square(ys-ys_pred), reduction_indices=[1])
        else:
            # SOFTMAX CROSS ENTROPY (COST FUNCTION)
            loss = -tf.reduce_sum(ys*tf.log(tf.clip_by_value(ys_pred, 1e-10, 1.0)))
            # loss = -tf.reduce_sum(ys*tf.log(ys_pred, 1e-10, 1.0), reduction_indices=[1])
            # loss = tf.nn.softmax_cross_entropy_with_logits(pred_outputs, outputs)          
            
        return loss    
    
    # CREATE MODEL
    @staticmethod
    def predict_model(X, parms, act_function): 
        W, bias = parms['weights'], parms['biases']
        act_fction, Out_fct = act_function['act_fction'], act_function['OutActi_fction']
        I, Ls = 1, {'layer%s'%ls:'' for ls in xrange(1, len(W)+1)}         
        while I < len(W): 
            # HISTOGRAM OF THE WEIGHTS AND BIASES
            tf.histogram_summary('layer%s '%(I)+'/Weights%s'%(I), W['w%s'%(I)])
            tf.histogram_summary('layer%s '%(I)+'/biases%s'%(I), bias['b%s'%(I)])
            if (I==1):
                Ls['layer%s'%(I)] = tf.add(tf.matmul(X, W['w%s'%(I)]), bias['b%s'%(I)])
                if act_fction is not None:
                    Ls['layer%s'%(I)] = act_fction(Ls['layer%s'%(I)])
                # CREATE A SAMMARY TO VISUALIZE THE FIRST LAYER RELU ACTIVATION
                tf.histogram_summary("activation_layer %s"%(I), Ls['layer%s'%(I)])
            else:
                Ls['layer%s'%(I)] = tf.add(tf.matmul(Ls['layer%s'%(I-1)], W['w%s'%(I)]), bias['b%s'%(I)])
                if act_fction is not None:
                    Ls['layer%s'%(I)] = act_fction(Ls['layer%s'%(I)])
                # CREATE A SAMMARY TO VISUALIZE THE FIRST LAYER RELU ACTIVATION
                tf.histogram_summary("activation_layer%s "%(I), Ls['layer%s'%(I)])
            I += 1   
             
        # COMPUTING THE OUTPUT LAYER VALUE  
        prediction = tf.add(tf.matmul(Ls['layer%s'%(I-1)], W['w%s'%(I)]), bias['b%s'%(I)])
        # CREATE A SAMMARY TO VISUALIZE THE FIRST LAYER RELU ACTIVATION
        tf.histogram_summary('layer%s '%I+'/Weights%s'%(I), W['w%s'%I])
        tf.histogram_summary('layer%s '%I+'/biases%s'%(I), bias['b%s'%I])
        if Out_fct is not None:
            prediction = Out_fct(prediction) 
                
        return prediction
    
    
    @staticmethod    
    def compute_accuracy(v_xs,v_ys, sess):
        # % OF CORRECT ANSWERS FOUND IN BATCH
        y_pred = sess.run(prediction, feed_dict = {xs: v_xs})
        is_correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        result = sess.run(accuracy,feed_dict = {xs: v_xs, ys: v_ys})
        
        return result
    
    
    
    
    
    
    
    
    
    
    
    
    
 


