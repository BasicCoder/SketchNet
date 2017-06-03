#!/usr/bin/env python
#-*- coding:utf-8 -*-

import tensorflow as tf 
import numpy as np 
import os 



# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([15, 15, 3, 64])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'wd1': tf.Variable(tf.random_normal([7, 7, 256, 512])), 
    'wd2': tf.Variable(tf.random_normal([1, 1, 512, 256])),
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

def ImageNetPos(_X, _weights = weights, _biases = biases,  dropout_prob = 1.0):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 256, 256, 3])

    # Layer 1
    with tf.name_scope('L1') as scope:
        # Convolution Layer 1
        conv1 = tf.nn.conv2d(_X,  _weights['wc1'], strides = [1, 3, 3, 1], padding = 'VALID', name = 'conv1')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, _biases['bc1']), name = 'relu1')
        # Max Pooling (down-sampling)
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

    # Layer 2
    with tf.name_scope('L2') as scope:
        # Convolution Layer 2
        conv2 = tf.nn.conv2d(pool1, _weights['wc2'], strides = [1, 1, 1, 1], padding='VALID', name='conv2')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, _biases['bc2']), name='relu2')
        # Max Pooling (down-sampling)
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

    # Layer 3
    with tf.name_scope('L3') as scope:
        # Convolution Layer 3
        conv3 = tf.nn.conv2d(pool2, _weights['wc3'], [1, 1, 1, 1], padding='SAME', name='conv3')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, _biases['bc3']), name='relu3')

    # Layer 4
    with tf.name_scope('L4') as scope:
        #Convolution Layer 4
        conv4 = tf.nn.conv2d(relu3, _weights['wc4'], [1, 1, 1, 1], padding='SAME', name='conv4')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, _biases['bc4']), name='relu4')


    # Layer 5
    with tf.name_scope('L5') as scope:
        # Convolution Layer 5
        conv5 = tf.nn.conv2d(relu4, _weights['wc5'], [1, 1, 1, 1], padding='SAME', name='conv5')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, _biases['bc5']), name='relu5')
        pool5 = tf.nn.max_pool(relu5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

    # Layer 6
    with tf.name_scope('L6') as scope:
        # Fully connected layer 1
        fc6 = tf.nn.conv2d(pool5, _weights['wd1'], [1, 1, 1, 1], padding='VALID', name='fc6')
        relu6 = tf.nn.relu(tf.nn.bias_add(fc6, _biases['bd1']), name='relu6')
        dropout6 = tf.nn.dropout(relu6, keep_prob=dropout_prob, name='dropout6')

    # Layer 7
    with tf.name_scope('L7') as scope:
        # Fully connected layer 2
        fc7 = tf.nn.conv2d(dropout6, _weights['wd2'], [1, 1, 1, 1], padding='VALID', name='fc7')
        relu7 = tf.nn.relu(tf.nn.bias_add(fc7, _biases['bd2']), name='relu7')
        # dropout7 = tf.nn.dropout(relu7, keep_prob=dropout_prob, name='dropout7')
        dense2 = tf.nn.l2_normalize(relu7, dim = 0, name='fc2')

    
    return dense2