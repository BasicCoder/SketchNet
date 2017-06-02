#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf 
import numpy as np 
import os

from ReadData import ReadData
from SketchNet import SketchNet
from ImageNetPos import ImageNetPos
from ImageNetNeg import ImageNetNeg


data_name = ''
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 20
margin = 50

def EuclideanDist(a, b):
    return tf.sqrt(tf.reduce_sum(tf.square(a - b), 2))

def run_training():
    s, ipos, ineg = ReadData()

    sketch_dense = SketchNet(s)
    image_pos_dense = ImageNetPos(ipos)
    image_neg_dense = ImageNetNeg(ineg)


    cost = max(0, margin + EuclideanDist(sketch_dense, image_pos_dense) - EuclideanDist(sketch_dense, image_neg_dense))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Add the variable initializer Op to the graph
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver()

    # Create a session for running the Ops on the graph
    with tf.Session() as sess:

        # Restore the variables or Run the Op to initialize variables
        latest_ckpt_file = tf.train.latest_checkpoint(os.path.join(FLAGS.logdir, 'ckpt'))
        if latest_ckpt_file is not None:
            saver.restore(sess, latest_ckpt_file)
            print('Model Restored')
        else:
            sess.run(init)

        

if __name__ == '__main__':
    run_training()
