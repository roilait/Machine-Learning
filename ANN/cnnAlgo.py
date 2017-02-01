# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 21:35:30 2017

@author: moussa
"""
import tensorflow as tf
# IMPORT CLASSES

class ConvNet(object):
    # PREPARING THE CONVOLUTION LAYER INPUTS
    @staticmethod
    def in_out_channels(in_channels, out_channels,filters):
        # shape = [[F,F,in_chls,out_chls], [F1,F1,out_chls,out_chls1], ......,]
        shapeList = [filters[i]+[in_channels[i], out_channels[i]] if (i==0) else 
                                       filters[i]+[out_channels[i-1], out_channels[i]] for i in xrange(len(filters))]
        return shapeList
    # WEIGHT VARIABLE FUNCTION
    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    # BIAS VARIBLE FUNCTION
    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
     
     
     
    # CONVOLUTION LAYER FUNCTION
    # stride[1, x_movement, y_movement, 1]
    # we must have strides[0] - strides[3] = 1    
    @staticmethod
    def fct_conv2d(X,W,s_x,s_y):
        return tf.nn.conv2d(X, W, strides=[1,s_x,s_y,1], padding='SAME')
    # MAX POOLING LAYER FUNCTION
    @staticmethod
    def max_pool_2x2(X,k_x,k_y):
        # stride[1, x_movment, y_movment, 1]
        return tf.nn.max_pool(X, ksize = [1,k_x,k_y, 1],strides=[1,k_x,k_y,1], padding='SAME')
 
    
    # PREPARING THE OUTPUT AND OUTPUT POOLING
    @staticmethod
    def layers(nbreConv):
        for i in xrange(nbreConv):
            out_conv, out_pool = {'y%s'%(i):''}, {'y%s'%(i):''}
        return out_conv, out_pool
        
    # DEFINE PLACEHOLDER FOR INPUTS TO NETWORK #
    def training(x_img, n_convLayer, shapeList):
        # PREPARING THE OUTPUTS LAYERS
        out_conv, out_pool = ConvNet.layers(n_convLayer)
        
        for conv in range(n_convLayer):
            # GET THE WEIGHT k
            W_conv = ConvNet.weight_variable(shapeList[conv])
            # GET THE BIAS k
            b_conv = ConvNet.bias_variable(shapeList[conv][-1])
            # GET THE x and y MOVMENT, STRIDE
            
            sx, sy, kx, ky = 1, 1, 2, 2            
            if (conv==0):                
                out_conv['y%s'%(conv)] = tf.nn.relu(ConvNet.conv2d(x_img, W_conv, sx, sy) + b_conv)
                out_pool['y%s'%(conv)] = ConvNet.max_pool_2x2(out_conv['y%s'%(conv)], kx, ky)
            else:
                out_conv['y%s'%(conv)] = tf.nn.relu(ConvNet.conv2d(out_pool['y%s'%(conv-1)], W_conv, sx, sy) + b_conv)
                out_pool['y%s'%(conv)] = ConvNet.max_pool_2x2(out_conv['y%s'%(conv)], kx, ky)
                
    K, L, M = 4, 8, 12
    input_channels = [1]
    output_channels = [K, L, M]
    inputs = tf.placeholder(tf.float32, shape=[None, 784])
    outputs = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(inputs, [-1,28,28,1])
    # CONV1 LAYER #
    K = 32
    W_conv1 = weight_variable([5,5,1,K]) # 5x5:filter size 5x5, 1:input channels,K: output channels 
    b_conv1 = bias_variable([K])
    h_conv1 = tf.nn.relu(fct_conv2d(x_image, W_conv1) + b_conv1)# output size 28x28x32
    h_pool1 = max_pool_2x2(h_conv1)
    

    
    
    

