# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:24:03 2016

@author: moussa
"""
import tensorflow as tf
import numpy as np
# IMPORT THE CLASSES HERE
import getData as getd

class CrossValidation(object):
    def __init__(self, Data, feats):
        cv_data, cv_labels = Data['train_x'], Data['train_y']
        if np.any(Data['test_x']): 
            cv_data = np.concatenate((cv_data, Data['test_x']), axis = 0)
            cv_labels = np.concatenate((cv_labels, Data['test_y']), axis = 0)
        if np.any(Data['valid_x']): 
            cv_data = np.concatenate((cv_data, Data['valid_x']), axis = 0)
            cv_labels = np.concatenate((cv_labels, Data['valid_y']), axis = 0)
            
        # THE CROSS VALIDATIOM METHOD    
        L, fold = len(cv_data), int(len(cv_data)/feats['k_fold'])
        n_in, n_out = len(cv_data[0]), len(cv_labels[0])
        
        # BUILD THE K-FOLDS FOR THE LEARNING
        xdata_fold = [cv_data[i:i + fold] for i in xrange(0, L, fold)]
        labels_fold = [cv_labels[j:j + fold] for j in xrange(0, L, fold)]
        
        CrossValidation.CV_model(xdata_fold, labels_fold, n_in, n_out, feats)

    @staticmethod
    def CV_model(xdata_fold, labels_fold, n_inputs, n_classes, features):
        epochs =features['TrainEpochs'] 
        TrBatchSize = features['TrainBatchSize'] 
        
        # NETWORK PARAMETERS 
        n_hidden_1 = features['hidden_layers'][0]
        n_hidden_2 = features['hidden_layers'][0]
        
        # DEFINE PLACEHOLDERS
        inputs = tf.placeholder(tf.float32, shape = [None, n_inputs], name = 'InputData')
        outputs = tf.placeholder(tf.float32, shape = [None, n_classes], name = 'LabelData')
        
        # DEFINE WEIGHTS AND BIASES OF THE NEURAL NETWORK    
        H = [n_inputs, n_hidden_1, n_hidden_2, n_classes]
        weights = {'w'+str(i+1):CrossValidation.init_weight([H[i],H[i+1]]) for i in xrange(len(H)-1)} 
        biases = {'b'+str(i+1):CrossValidation.init_biases([H[i+1]]) for i in xrange(len(H)-1)}         
        '''        
        weights = {'w1':CrossValidation.init_weight([n_inputs, n_hidden_1]),
                   'w2':CrossValidation.init_weight([n_hidden_1, n_hidden_2]),                    
                   'w3':CrossValidation.init_weight([n_hidden_2, n_classes])
                  }                  
        biases = {'b1':tf.Variable(tf.random_normal([n_hidden_1]), name = 'b1'),
                  'b2':tf.Variable(tf.random_normal([n_hidden_2]), name = 'b2'),
                  'b3':tf.Variable(tf.random_normal([n_classes]), name = 'b3')
                 }  
        '''        
        # BUILD MODEL
        output_layer = CrossValidation.net_model(inputs, weights, biases) 
        
        # SOFTMAX CROSS ENTROPY (COST FUNCTION)
        loss = -tf.reduce_sum(outputs*tf.log(tf.clip_by_value(output_layer, 1e-10, 1.0)))
        # loss = tf.nn.softmax_cross_entropy_with_logits(pred_outputs, outputs)
        # SQUARED LOSS FUNCTION
        # loss = tf.square(output_layer-outputs) 
        loss = tf.reduce_mean(loss)
            
        # SET THE OPTIMIZER TO UPDATE THE HYPER-PARAMTERS, i.e, OUR BACKPROPAGATION ALGORITHM
        # optimizer = tf.train.AdamOptimizer(learning_rate=features['learning_rate'])
        optimizer = tf.train.GradientDescentOptimizer(features['learning_rate'])
        train = optimizer.minimize(loss) 
        
        # % OF CORRECT ANSWERS FOUND IN BATCH
        is_correct = tf.equal(tf.argmax(outputs, 1), tf.argmax(output_layer, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        
        # AFTER DEFINING OUR NEURAL NETWORK, LET'S TO INITIALIZE ALL THE VARIABLES
        init = tf.initialize_all_variables()  
        
        # NOW LET US CREATE A SESSION, AND RUN OUR NEURAL NETWORK SESSION
        with tf.Session() as sess:
            # CREATE INITIALIZED VARIABLE
            sess.run(init)
            
            fold_avg_error = np.zeros((len(xdata_fold), 1))
            for item in xrange(len(xdata_fold)):
                Test_x, Test_y = xdata_fold[item], labels_fold[item]
                data_x = [xdata_fold[i] for i in xrange(len(xdata_fold)) if i!=item]
                data_y = [labels_fold[j] for j in xrange(len(xdata_fold)) if j!=item] 
            
                # DATA CONCATENATION
                Train_x, Train_y = data_x[0], data_y[0]
                for i in xrange(1, len(data_x)):
                    Train_x = np.concatenate((Train_x , data_x[i]), axis = 0)
                    Train_y = np.concatenate((Train_y , data_y[i]), axis = 0)
                
                # CROSS VALIDATION MODEL 
                #fold_avg_error[item][0] = self.CV_model(n_in, n_out, feats) 
            
                # LOOP OVER ALL BATCHES              
                batches_x, batches_y = getd.Batches.get_batches(Train_x, Train_y, TrBatchSize)
                epoch, train_cost = 0, []
                while epoch < epochs:
                    avg_cost, avg_acc = 0., 0.
                    for batch_x, batch_y in zip(batches_x, batches_y):
                        train_dict = {inputs: batch_x, outputs: batch_y}
                        _, cost, acc = sess.run([train, loss, accuracy], feed_dict = train_dict)
                        avg_acc += acc
                        avg_cost += cost/len(batches_x)
                    train_cost.append(avg_cost)
                
                    # DISPLAY LOGS PER EPOCH STEP
                    if (epoch + 1) % features['displayStep']==0:
                        print ("epoch:", '%04d' % (epoch + 1), "cost=", "{:.3f}".format(avg_cost)) 
                    epoch += 1
                
                # EVALUATING THE MODEL
                test_dict = {inputs: Test_x, outputs: Test_y}
                test_cost, test_accu = sess.run([loss, accuracy], feed_dict = test_dict)                 
                
                fold_avg_error[item][0] = test_cost
        print fold_avg_error        
        #return test_cost       
 
 
    # WEIGHT AND BIASES INITIALIZATION
    @staticmethod
    def init_weight(shape):
        return tf.Variable(tf.random_normal(shape, stddev = 0.01))
    def init_biases(shape):
        return tf.Variable(tf.random_normal(shape))

    # CREATE MODEL
    @staticmethod
    def net_model(X, weights, biases):
        # NOW CREATE OUR NEURAL NETWORKS COMPUTIONAL GRAPH
        hidden_layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
        hidden_layer_1 = tf.nn.relu(hidden_layer_1)
        # CREATE A SAMMARY TO VISUALIZE THE FIRST LAYER RELU ACTIVATION
        tf.histogram_summary("relu1", hidden_layer_1)
        # NEXT HIDDEN LAYER
        hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, weights['w2']), biases['b2'])
        hidden_layer_2 = tf.nn.relu(hidden_layer_2)
        # CREATE A SAMMARY TO VISUALIZE THE FIRST LAYER RELU ACTIVATION
        tf.histogram_summary("relu2", hidden_layer_2)
        # COMPUTE THE OUTPUT LAYER
        output_layer = tf.add(tf.matmul(hidden_layer_2, weights['w3']), biases['b3']) 
        output_layer = tf.nn.softmax(output_layer) 
        
        return output_layer