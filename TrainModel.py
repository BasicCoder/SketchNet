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
learning_rate_init = 0.01
training_iters = 135 * 500
batch_size = 135
display_step = 5
save_step = 200
margin = 10.0
dropout = 0.9

dir_name = r'./CheckPoin/'

# Store layers weight & bias
with tf.name_scope('Image_Weights') as scope:
    image_weights = {
        'wc1': tf.Variable(tf.random_normal([15, 15, 3, 64]), name = 'wc1'),
        'wc2': tf.Variable(tf.random_normal([5, 5, 64, 128]), name = 'wc2'),
        'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256]), name = 'wc3'),
        'wc4': tf.Variable(tf.random_normal([3, 3, 256, 256]), name = 'wc4'),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256]), name = 'wc5'),
        'wd1': tf.Variable(tf.random_normal([8*8*256, 512]), name = 'wd1'), 
        'wd2': tf.Variable(tf.random_normal([512, 256]), name = 'wd2'),
    }
with tf.name_scope('Image_Biases') as scope:
    image_biases = {
        'bc1': tf.Variable(tf.random_normal([64]), name = 'bc1'),
        'bc2': tf.Variable(tf.random_normal([128]), name = 'bc2'),
        'bc3': tf.Variable(tf.random_normal([256]), name = 'bc3'),
        'bc4': tf.Variable(tf.random_normal([256]), name = 'bc4'),
        'bc5': tf.Variable(tf.random_normal([256]), name = 'bc5'),
        'bd1': tf.Variable(tf.random_normal([512]), name = 'bd1'),
        'bd2': tf.Variable(tf.random_normal([256]), name = 'bd2'),
    }

with tf.name_scope('Sketch_Weights') as scope:
    sketch_weights = {
        'wc1': tf.Variable(tf.random_normal([15, 15, 3, 64]), name = 'wc1'),
        'wc2': tf.Variable(tf.random_normal([5, 5, 64, 128]), name = 'wc2'),
        'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256]), name = 'wc3'),
        'wc4': tf.Variable(tf.random_normal([3, 3, 256, 256]), name = 'wc4'),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256]), name = 'wc5'),
        'wd1': tf.Variable(tf.random_normal([8*8*256, 512]), name = 'wd1'), 
        'wd2': tf.Variable(tf.random_normal([512, 256]), name = 'wd2'),
    }
with tf.name_scope('Sketch_Biases') as scope:
    sketch_biases = {
        'bc1': tf.Variable(tf.random_normal([64]), name = 'bc1'),
        'bc2': tf.Variable(tf.random_normal([128]), name = 'bc2'),
        'bc3': tf.Variable(tf.random_normal([256]), name = 'bc3'),
        'bc4': tf.Variable(tf.random_normal([256]), name = 'bc4'),
        'bc5': tf.Variable(tf.random_normal([256]), name = 'bc5'),
        'bd1': tf.Variable(tf.random_normal([512]), name = 'bd1'),
        'bd2': tf.Variable(tf.random_normal([256]), name = 'bd2'),
    }


def EuclideanDist(a, b):
    return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(a, b)), 1))

def run_training():
    

    sketchs_placeholder = tf.placeholder(tf.float32)
    images_neg_placeholder = tf.placeholder(tf.float32)
    images_pos_placeholder = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)

    # Three Branch Net
    sketch_dense = SketchNet(sketchs_placeholder, _weights = sketch_weights, _biases = sketch_biases, dropout_prob = keep_prob)
    image_pos_dense = ImageNetPos(images_neg_placeholder, _weights = image_weights, _biases = image_biases, dropout_prob = keep_prob)
    image_neg_dense = ImageNetNeg(images_pos_placeholder, _weights = image_weights, _biases = image_biases, dropout_prob = keep_prob)
    tf.summary.tensor_summary("sketch_dense", sketch_dense)
    tf.summary.tensor_summary("image_pos_dense", image_pos_dense)
    tf.summary.tensor_summary("image_neg_dense", image_neg_dense)

    # Euclidean Distance
    dist_pos = EuclideanDist(sketch_dense, image_pos_dense)
    dist_neg = EuclideanDist(sketch_dense, image_neg_dense)
    margins = tf.constant(margin, dtype = tf.float32, shape = [batch_size])
    print(dist_pos, dist_neg, margins)

    with tf.name_scope('Loss') as scope:
        zeros = tf.constant(0.0, dtype = tf.float32, shape = [batch_size])
        cost = tf.reduce_sum( tf.nn.relu(margins + dist_pos - dist_neg) )
        tf.summary.scalar("loss", cost)
    
    with tf.name_scope('Optimizer') as scope:
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, 100, 0.98, staircase = True, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost, global_step = global_step)
        tf.summary.scalar('global step', global_step)
        tf.summary.scalar('learning_rate', learning_rate)


    # Add the variable initializer Op to the graph
    init = tf.global_variables_initializer()

    # Merge all summary
    merged_summary_op = tf.summary.merge_all()

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
        
        summary_writer = tf.summary.FileWriter('./logs', graph_def=sess.graph_def)

        step = 1
        dataset = ReadData(sess, batch_size, is_train = True)
        while step * batch_size <= training_iters:
            s, ipos, ineg = next(dataset)

            print('Start optimizer :', step)
            sess.run(optimizer, feed_dict = {sketchs_placeholder : s, images_neg_placeholder : ipos, 
                                            images_pos_placeholder : ineg, keep_prob: dropout})
            print('optimizer :', step, 'finised!')

            if step % display_step == 0:
                summary_str, loss = sess.run([merged_summary_op, cost], feed_dict = {sketchs_placeholder : s, images_neg_placeholder : ipos, 
                                            images_pos_placeholder : ineg, keep_prob: 1.0})
                print("Iter" + str(step) + ", Minibatch Loss= " + "{:.09f}".format(loss))
                
                summary_writer.add_summary(summary_str, step)
            
            # Save Model
            if step % save_step == 0:
                print("Saving model checkpoint after {} steps.".format(step))
                checkpoint_file = os.path.join(dir_name, 'ckpt', 'model.ckpt')
                saver.save(sess, checkpoint_file, step)
                print('Checkpoint Saved!')
            step += 1
        
        print("Optimization Finished!")

if __name__ == '__main__':
    run_training()
