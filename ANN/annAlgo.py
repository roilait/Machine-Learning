#!/usr/bin/python
# -*- coding: utf-8 -*-
# http://jorditorres.org/first-contact-with-tensorflow/
# https://www.analyticsvidhya.com/blog/2016/10/an-introduction-to-implementing-neural-networks-using-tensorflow/
# https://www.youtube.com/watch?v=vq2nnJ4g6N0
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_advanced.py
# http://camron.xyz/index.php/2016/10/03/nutshell-all-the-models/
# https://www.youtube.com/watch?v=8XKSQTCzXEI&list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f&index=11
# https://www.youtube.com/watch?v=G5RY_SUJih4

#https://bioinformatics.ca/resources/jobs/tenure-track-position-machine-learning-data-mining-cloud-computing-and-big-data

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
        
        
                        
                    
                    
        # -----------------------------
        # -- CLASSIFICATION FUNCTION --
        # -----------------------------
        '''
        def classification(init):
            # NOW LET US CREATE A SESSION, AND RUN OUR NEURAL NETWORK SESSION
            with tf.Session() as sess:
                # CREATE INITIALIZED VARIABLE
                sess.run(init) 
                # DELET THE logs FILE, BEFORE TO ADD THE GRAPH
                getd.DeleteFileOrFolder(logs_path+'logs/')
                # SUMMARY WRITER GOES HERE
                train_writer = tf.train.SummaryWriter(logs_path+'logs/train', sess.graph)
                test_writer = tf.train.SummaryWriter(logs_path+'logs/test') 
                # PARAMETERS INITIALIZATION                       
                train_cost, train_accuracy = [], [] 
                test_cost, test_accuracy = [], [] 
                epoch = 0
                while epoch < train_epochs:
                    avg_cost, avg_acc = 0., 0.
                    # LOOP OVER ALL BATCHES
                    iters, i = 0, 0
                    while iters<len(x_data):
                        start, end = iters, iters + batchSize
                        batch_x, batch_y = x_data[start:end], y_data[start:end]
                        _, cost = sess.run([train, loss], feed_dict = {xs: batch_x, ys: batch_y})
                        
                        train_result = sess.run(merged, feed_dict = {xs: batch_x, ys: batch_y})
                        train_writer.add_summary(train_result, epoch)
                        
                        iters += batchSize
                    #if(epoch + 1) % displayStep==0:
                    print "====== FIN ========"
                    #sess.run(train, feed_dict = {xs: x_data, ys: y_data})
                    #if step%displayStep==0:
                    #train_result = sess.run(merged, feed_dict = {xs: x_data, ys: y_data})
                    # train_writer.add_summary(train_result, step)
                    
                    #train_result, cost = sess.run([merged, loss], feed_dict={xs: x_data,ys: y_data})
                    #train_writer.add_summary(train_result, epoch)
                    #else:
                    #    _, cost  = sess.run([train, loss], feed_dict={xs:batch_x,ys:batch_y})
                    ## WRITE LOGS AT EVERY ITERATION
                            
                    #avg_cost += cost/total_batch
                    #print 'fin===========', 
                     #   iters += batchSize
                      #  i += 1
                    
                    #train_cost.append(avg_cost)
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.3f}".format(cost))
                    if x_test is not None:
                        if(epoch + 1) % displayStep==0:
                            test_result = sess.run(merged, feed_dict = {xs: x_test, ys: y_test})                    
                            test_writer.add_summary(test_result, epoch)                
                        
                    epoch += 1
        '''            
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
        '''
        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary_scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary_scalar('stddev', stddev)
                tf.summary_scalar('max', tf.reduce_max(var))
                tf.summary_scalar('min', tf.reduce_min(var))
                tf.summary_histogram('histogram', var)
         
        ''' 
        
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
    
    
    
    
    
    
    
    
    
    
    
    
    
 





   
class Cls_model(object):    
    def __init__(self,Data, feats, hidden_layers):
        tr_cost, te_cost, tr_accu, te_accu = Cls_model.neural_network_model(Data, feats, hidden_layers)
        plt.figure()
        plt.plot(tr_cost)
        plt.plot(te_cost, 'r')
        
        plt.figure()
        plt.plot(tr_accu)
        plt.plot(te_accu, 'r')
        plt.grid(True)
        plt.ylim((0,2))
        plt.show()
        
    @staticmethod   
    def neural_network_model(Data, feats, hidden_layers,selectModel = False, HL = []):
        Train_x, Train_y = Data['train_x'], Data['train_y'] 
        if selectModel:
            Test_x, Test_y = Data['test_x'], Data['test_y'] 
            # Valid_x, Valid_y = Data['valid_x'], Data['valid_y'] 
        
        # SET REMAINING VARIABLES
        training_epochs = feats['TrainEpochs']
        batchSize = feats['TrainBatchSize']
        learning_rate = feats['learning_rate']
        displayStep = feats['displayStep']
        
        # NETWORK PARAMETERS 
        n_inputs = len(Train_x[0]) 
        n_classes = len(Train_y[0])
        # PREPARING THE LAYERS LIST
        HL.append(n_inputs)
        HL.extend(hidden_layers)
        HL.append(n_classes)
        
        # DEFINE PLACEHOLDERS
        inputs = tf.placeholder(tf.float32, [None, n_inputs])
        outputs = tf.placeholder(tf.float32, [None, n_classes])
        
        # DEFINE WEIGHTS AND BIASES OF THE NEURAL NETWORK
        weights, biases = ModelClasses.get_weights(HL), ModelClasses.get_biases(HL)               
        # BUILD MODEL
        parms = {'weights':weights,'biases':biases}
        activation_function = {'act_fction':tf.nn.relu, 'OutActi_fction':tf.nn.softmax}
        prediction = ModelClasses.predict_model(inputs, parms, activation_function)              
        
        # COST FUNCTION
        loss = ModelClasses.cost_function(outputs, prediction)
        loss = tf.reduce_mean(loss)
        # CREATE A SUMMARY TO MONOTOR COST TENSOR
        tf.scalar_summary("Loss", loss)
            
        # SET THE OPTIMIZER TO UPDATE THE HYPER-PARAMTERS, i.e, OUR BACKPROPAGATION ALGORITHM
        # optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)
        
        # % OF CORRECT ANSWERS FOUND IN BATCH
        is_correct = tf.equal(tf.argmax(outputs, 1), tf.argmax(prediction, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        
        # AFTER DEFINING OUR NEURAL NETWORK, LET'S TO INITIALIZE ALL THE VARIABLES
        init = tf.initialize_all_variables()
        
        # NOW LET US CREATE A SESSION, AND RUN OUR NEURAL NETWORK SESSION
        batches_x, batches_y = ModelClasses.get_batches(Train_x, Train_y, batchSize)
        total_batches = len(batches_x)
        with tf.Session() as sess:
            # CREATE INITIALIZED VARIABLE
            sess.run(init)                        
            epoch, train_cost, train_accu, test_cost, test_accu = 0, [], [], [], []       
            while epoch < training_epochs:
                avg_cost, avg_acc = 0., 0.
                # LOOP OVER ALL BATCHES
                for batch_x, batch_y in zip(batches_x, batches_y):
                    train_data = {inputs:batch_x, outputs:batch_y}
                    _, cost, acc = sess.run([train, loss, accuracy], feed_dict = train_data)
                    avg_acc += acc
                    avg_cost += cost
                train_cost.append(avg_cost/total_batches)
                train_accu.append(avg_acc/total_batches) 
                                
                # DISPLAY LOGS PER EPOCH STEP
                if (epoch + 1) % displayStep==0:
                    print ("epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))                    
                    
                # EVALUATING THE MODEL
                if selectModel:
                    # tes_cost, tes_acc = 0., 0.
                    # for xx, yy in zip(batches_xx, batches_yy):
                    test_data = {inputs:Test_x, outputs:Test_y}
                    te_cost, accu = sess.run([loss, accuracy], feed_dict = test_data)
                    # tes_cost += te_cost
                    # tes_acc += accu
                    test_cost.append(te_cost)
                    test_accu.append(acc)
                   
                epoch += 1
                          
        return train_cost, test_cost, train_accu, test_accu
        

        
        
        
  





      
        
class Tfclassification(object):    
    def __init__(self,input_data, feats, vectorize=True):
        train_cost, test_cost = self.NetTensorflow(input_data, feats)
        
        plt.figure(1)
        plt.plot(train_cost)
        #plt.plot(test_cost, 'r')
        plt.show()
  
    # CREATE MODEL
    @staticmethod
    def net_model(X, weights, biases):
        # NOW CREATE OUR NEURAL NETWORKS COMPUTIONAL GRAPH
        # HIDDEN LAYER WITH RELU ACTIVATION
        hidden_layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
        hidden_layer_1 = tf.nn.relu(hidden_layer_1)
        # CREATE A SAMMARY TO VISUALIZE THE FIRST LAYER RELU ACTIVATION
        tf.histogram_summary("relu1", hidden_layer_1)        
        # HIDDEN LAYER WITH RELU ACTIVATION
        hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, weights['w2']), biases['b2'])
        hidden_layer_2 = tf.nn.relu(hidden_layer_2)
        # CREATE A SAMMARY TO VISUALIZE THE FIRST LAYER RELU ACTIVATION
        tf.histogram_summary("relu2", hidden_layer_2)
        # OUTPUT LAYER
        output_layer = tf.add(tf.matmul(hidden_layer_2, weights['w3']), biases['b3']) 
        output_layer = tf.nn.softmax(output_layer) 
        
        return output_layer
        
    def NetTensorflow(self, Data, feats):         
        Train_x, Train_y = Data['train_x'], Data['train_y'] 
        Test_x, Test_y = Data['test_x'], Data['test_y'] 
        # Valid_x, Valid_y = Data['valid_x'], Data['valid_y'] 
        
        # NETWORK PARAMETERS 
        n_inputs = len(Train_x[0]) 
        n_hidden_1 = feats['hidden_layers'][0]
        n_hidden_2 = feats['hidden_layers'][0]
        n_classes = len(Train_y[0]) 
        
        # SET REMAINING VARIABLES
        # te_batch_size = feats['TestBatchSize']
        learning_rate = feats['learning_rate']  
        
        # DEFINE THE BATCHES
        batches_x, batches_y = getd.Batches.get_batches(Train_x, Train_y, feats['TrainBatchSize'])
        total_batch = len(batches_x)
        
        # DEFINE PLACEHOLDERS
        inputs = tf.placeholder(tf.float32, [None, n_inputs], name = 'InputData')
        outputs = tf.placeholder(tf.float32, [None, n_classes], name = 'LabelData')

        # DEFINE WEIGHTS AND BIASES OF THE NEURAL NETWORK
        H = [n_inputs, n_hidden_1, n_hidden_2]
         
                    
        weights = {'w1':tf.Variable(tf.random_normal([n_inputs, n_hidden_1]), name = 'w1'),
                   'w2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name = 'w2'),                    
                   'w3':tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name = 'w3')
                  }
                  
        biases = {'b1':tf.Variable(tf.random_normal([n_hidden_1]), name = 'b1'),
                  'b2':tf.Variable(tf.random_normal([n_hidden_2]), name = 'b2'),
                  'b3':tf.Variable(tf.random_normal([n_classes]), name = 'b3')
                 }
                 
        # ENCAPSULATING ALL OPS INTO SCOPES, MAKING TENSORBOARD'S GRAPH
        with tf.name_scope('Model'):
            # BUILD MODEL
            pred_outputs = Tfclassification.net_model(inputs, weights, biases)            
        
        with tf.name_scope('Loss'):
            # SOFTMAX CROSS ENTROPY (COST FUNCTION)
            #loss = -tf.reduce_sum(outputs*tf.log(tf.clip_by_value(pred_outputs, 1e-10,1.0)))
            loss = tf.nn.softmax_cross_entropy_with_logits(pred_outputs, outputs)
            # SQUARED LOSS FUNCTION
            # loss = tf.square(output_layer-outputs) 
            loss = tf.reduce_mean(loss)
            
        with tf.name_scope('SDG'):
            # GRADIENT DESCENT
            #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            # Op TO CALCULATE EVERY VARIABLE GRADIENT
            grads = tf.gradients(loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            # OP TO UPDATE ALL VARIABLE ACCORDING TO THEIR GRADIENT
            apply_grads = optimizer.apply_gradients(grads_and_vars=grads)
             
        with tf.name_scope('Accuracy'):
            # COMPUTE THE ACCURACY
            acc = tf.equal(tf.argmax(pred_outputs,1), tf.argmax(outputs,1))
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))
            
        # AFTER DEFINING OUR NEURAL NETWORK, LET'S TO INITIALIZE ALL THE VARIABLES
        init = tf.initialize_all_variables()
        
        # CREATE A SUMMARY TO MONOTOR COST TENSOR
        tf.scalar_summary("loss", loss)
        
        # CREATE A SUMMARY TO MONITOR ACCURACY TENSOR
        tf.scalar_summary("accuracy", acc)
        
        # CREATE SUMMARIES TO VISUALIZE WEIGHTS
        for var in tf.trainable_variables():
            tf.histogram_summary(var.name, var)
            
        # SUMMARIZE ALL GRADIENTS
        for grad, var in grads:
            tf.histogram_summary(var.name + '/gradient', grad)
            
        # MERGE ALL SUMMARIES INTO A SINGLE OP
        merged_summary_op = tf.merge_all_summaries()
        
        # LAUNCH THE GRAPH
        with tf.Session() as sess:
            # CREATE INITIALIZED VARIABLE
            sess.run(init)            
            # OP TO WRITE LOGS TO TENSORBOARD
            summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())                          
            epoch, train_cost, test_cost = 0, [], []            
            # TRAINING CYCLE
            while epoch < feats['TrainEpochs'] :
                avg_cost = 0.                
                # LOOP OVER ALL BATCHES
                i = 0
                for batch_x, batch_y in zip(batches_x, batches_y):
                    # RUN OPTIMIZATION OP (BACKPROPAGATION) COST OP (TO GET LOSS VALUE) AND SUMMARY NODES
                    #print batch_y
                    #print '=================='
                    _, cost, summary = sess.run([apply_grads, loss, merged_summary_op], 
                                                      feed_dict = {inputs: batch_x, outputs: batch_y})
                    # WRITE LOGS AT VERY ITERATION
                    summary_writer.add_summary(summary, epoch*total_batch + i)
                    
                    # Write logs at every iteration
                    print ("hidden layer 1", sess.run(weights['w1']), sess.run(biases['b1']))
                    print ('+++++++++++++++++++++++++++++++++++++++++++')
                    print ("hidden layer 2", sess.run(weights['w2']), sess.run(biases['b2']))
                    print ('------------------------------------------------')
                    print ("output layer", sess.run(weights['w3']), sess.run(biases['b3']))
                    print ('================================================')
                    # COMPUTE AVERAGE LOSS
                    avg_cost += cost/total_batch
                    print 
                    i += 1                 
                train_cost.append(avg_cost)
                # DISPLAY LOGS PER EPOCH STEP
                #if (epoch + 1) % feats['displayStep']==0:
                    #print ("epoch1:", '%04d' % (epoch + 1), "cost=", "{:.3f}".format(avg_cost))
                # print Test_x[0]
                '''
                if Test_x:
                    avg_loss = 0.
                    m = len(Test_x)
                    te_batches_x = [Test_x[i:i + te_batch_size] for i in xrange(0, m, te_batch_size)]
                    te_batches_y = [Test_y[j:j + te_batch_size] for j in xrange(0, m, te_batch_size)]
                    for data_x, data_y in zip(te_batches_x, te_batches_y):
                        y, te_loss = sess.run([pred_outputs, loss],{inputs:data_x, outputs: data_y})
                        avg_loss += te_loss/len(Test_x)
                        print ('test', y, '-', data_y)
                    test_cost.append(avg_loss)
                '''
                epoch += 1
                        
            #print("Optimization Finished!")              
        return train_cost, 0 #test_cost #avg_tr_cost, avg_test_cost, avg_accuracy
                          




       
    
