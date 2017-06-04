#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf 
import numpy as np 
import os

from ReadData_np import ReadData
from SketchNet import SketchNet
from ImageNetPos import ImageNetPos
from ImageNetNeg import ImageNetNeg


data_name = ''
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 20
margin = 50.0
dropout = 0.8

dir_name = r'./CheckPoin/'

def EuclideanDist(a, b):
    return tf.sqrt(tf.reduce_sum(tf.square(a - b), 1))

def run_training():
    
    sketchs_placeholder = tf.placeholder(tf.float32)
    images_neg_placeholder = tf.placeholder(tf.float32)
    images_pos_placeholder = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)

    sketch_dense = SketchNet(sketchs_placeholder, dropout_prob = keep_prob)
    image_pos_dense = ImageNetPos(images_neg_placeholder, dropout_prob = keep_prob)
    image_neg_dense = ImageNetNeg(images_pos_placeholder, dropout_prob = keep_prob)

    margins = tf.constant(margin, dtype = tf.float32, shape = [batch_size, 256])
    dist_pos = EuclideanDist(sketch_dense, image_pos_dense)
    dist_neg = EuclideanDist(sketch_dense, image_neg_dense)
    cost = tf.reduce_sum( tf.nn.relu(margins + dist_pos - dist_neg) )


    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Add the variable initializer Op to the graph
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver()

    # Create a session for running the Ops on the graph
    with tf.Session() as sess:

        # Restore the variables or Run the Op to initialize variables
        latest_ckpt_file = tf.train.latest_checkpoint(os.path.join(dir_name, 'ckpt'))
        if latest_ckpt_file is not None:
            saver.restore(sess, latest_ckpt_file)
            print('Model Restored')
        else:
            sess.run(init)
        
        step = 1
        dataset = ReadData(sess, batch_size)
        while step * batch_size < training_iters:
            s, ipos, ineg = next(dataset)

            print('Start optimizer :', step)
            sess.run(optimizer, feed_dict = {sketchs_placeholder : s, images_neg_placeholder : ipos, 
                                            images_pos_placeholder : ineg, keep_prob: dropout})
            print('optimizer :', step, 'finised!')
            if step * display_step == 0:
                loss = sess.run(cost, feed_dict = {sketchs_placeholder : s, images_neg_placeholder : ipos, 
                                            images_pos_placeholder : ineg, keep_prob: 1.0})
                print("Iter" + str(step) + ", Minibatch Loss= " + "{:.06f}".format(loss))

            step += 1
        
        print("Optimization Finished!")

if __name__ == '__main__':
    run_training()
