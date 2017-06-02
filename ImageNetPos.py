#!/usr/bin/env python
#-*- coding:utf-8 -*-

import tensorflow as tf 
import numpy as np 
import os 


def conv2d(name, l_input, w, b, step):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, step, step, 1], padding='SAME'),b), name=name)

def max_pool(name, l_input, k, step):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([15, 15, 1, 64])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'wd1': tf.Variable(tf.random_normal([256, 512])), 
    'wd2': tf.Variable(tf.random_normal([512, 256])),
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bc4': tf.Variable(tf.random_normal([256])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([512])),
    'bd2': tf.Variable(tf.random_normal([256])),
}

def ImageNetPos(_X, _weights = weights, _biases = biases, _dropout = 0.8):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 256, 256, 3])

    # Convolution Layer 1
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'], step = 3)
    # Max Pooling (down-sampling)
    pool1 = max_pool('pool1', conv1, k = 3, step = 2)

    # Convolution Layer 2
    conv2 = conv2d('conv2', pool1, _weights['wc2'], _biases['bc2'], step = 3)
    # Max Pooling (down-sampling)
    pool2 = max_pool('pool2', conv2, k = 3, step = 2)


    # Convolution Layer 3
    conv3 = conv2d('conv3', pool2, _weights['wc3'], _biases['bc3'], step = 1)


    #Convolution Layer 4
    conv4 = conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'], step = 1)



    # Convolution Layer 5
    conv5 = conv2d('conv5', conv4, _weights['wc5'], _biases['bc5'], step = 1)
    # Max Pooling (down-sampling)
    pool5 = max_pool('pool5', conv5, k = 3, step = 2)


    # Fully connected layer1
    dense1 = tf.reshape(pool5, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv5 output to fit dense layer input
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') # Relu activation

    # Fully connected layer2
    dense2 = tf.nn.l2_normalize(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], dim = 0, name='fc2') # Relu activation

    
    return dense2
