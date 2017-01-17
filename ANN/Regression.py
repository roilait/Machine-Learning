# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:53:11 2016

@author: moussa
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class TensorflowRegression(object):
    @staticmethod
    def regression():
        # PARAMETERS
        learning_rate = 0.1
        training_epochs = 1000
        display_step = 50
    
        # TRAINING DATA
        train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,
                          2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
        train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,
                          1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
        n_samples = train_X.shape[0]
        # TF GRAPH INPUT
        inputs = tf.placeholder("float")
        outputs = tf.placeholder("float")
    
        # SET MODEL WEIGHTS
        weights = tf.Variable(np.random.randn(), name='weight')
        biases = tf.Variable(np.random.randn(), name = 'biases')
    
        # CONSTRUCT A LINEAR MODEL
        pred = tf.add(tf.mul(inputs, weights), biases)
    
        # MEAN SQUARED ERROR
        cost = tf.reduce_sum(tf.pow(pred-outputs, 2))/(2*n_samples)
        # GRADIENT DESCENT
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        # INITIALIZING THE VARIABLES
        init = tf.initialize_all_variables()
    
        # LUNCHthe GRAPH
        with tf.Session() as sess:
            sess.run(init)
            # FIT ALL TRAINING DATA
            for epoch in xrange(training_epochs):
                for (x,y) in zip(train_X, train_Y):
                    sess.run(optimizer, feed_dict={inputs:x, outputs:y})
                
                #DISPLAY LOGS PER EPOCH STEP
                if (epoch + 1) % display_step==0:
                    c = sess.run(cost, feed_dict={inputs:train_X, outputs: train_Y})
                    print ("Epoche:", '%04d' % (epoch+1), "cost=", "{:.9}".format(c),
                       "w=", sess.run(weights), "b=", sess.run(biases))
            print ("Optimization Finished!")
            training_cost = sess.run(cost, feed_dict={inputs:train_X, outputs: train_Y})
            print("Training cost=", training_cost, "w=", sess.run(weights), "b=", sess.run(biases),'\n')
        
            # GRAPH DISPLAY
            plt.plot(train_X, train_Y, 'ro', label='Orignal data')
            plt.plot(train_X, sess.run(weights)*train_X + sess.run(biases), label='Fitted line')
            plt.legend()
            plt.show()
        
            # TESTING EXAMPLE< AS REQUESTED (ISSUE #2)
            test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
            test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
        
            print("Testing... (Mean square loss Comparison")
            testing_cost = sess.run(
                        tf.reduce_sum(tf.pow(pred-outputs,2))/(test_X.shape[0]),
                        feed_dict = {inputs:test_X, outputs:test_Y}) # same function as cost above
            print ("Testing cost=", testing_cost)
            print ("Absolute mean square loss difference:", 
                               abs(training_cost-testing_cost))
                               
            plt.plot(test_X, test_Y, 'bo', label="Testing data")
            plt.plot(train_X, sess.run(weights)*train_X + sess.run(biases), 
                                                           label='Fitted line')




  
class TensorflowRegre(object):
    def regression(self):
        num_points = 10000
        vectors_set = []
        for i in xrange(num_points):
            x1= np.random.normal(0.0, 0.55)
            y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
            vectors_set.append([x1, y1])

        x_data = [v[0] for v in vectors_set]
        y_data = [v[1] for v in vectors_set]

        num_points, vectors_set = 1000, []
        for i in xrange(num_points):
            x1= np.random.normal(0.1, 0.55)
            y1= x1 * 0.1 + 0.3 + np.random.normal(0.01, 0.03)
            vectors_set.append([x1, y1])

        x_data = np.array([v[0] for v in vectors_set])[:,np.newaxis]
        y_data = np.array([v[1] for v in vectors_set])[:,np.newaxis]
        
        #Graphic display
        plt.plot(x_data, y_data, 'ro')
        plt.legend()
        plt.show()

        W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
        b = tf.Variable(tf.zeros([1]))
        y = W * x_data + b

        loss = tf.reduce_mean(tf.square(y - y_data))
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        train = optimizer.minimize(loss)

        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)

        for step in xrange(100):
            sess.run(train)
            print(step, sess.run(W), sess.run(b))
            print(step, sess.run(loss))

            #Graphic display
            plt.plot(x_data, y_data, 'ro')
            plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
            plt.xlabel('x')
            plt.xlim(-2,2)
            plt.ylim(0.1,0.6)
            plt.ylabel('y')
            plt.legend()
            plt.show()
      
            



