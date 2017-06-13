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
training_iters = 135 * 1000
batch_size = 135
display_step = 5
save_step = 200
test_step = 50
margin = 6.0 / 407
dropout = 0.8
beta = 1e-5

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

def Count(a, b):
    count = 0
    for i in range(45):
        if a[i] < b[i]:
            count += 1
    return count

def Test(pos_val, neg_val):
    count = 0    
    count1 = Count(pos_val[0:45], neg_val[0:45])
    count2 = Count(pos_val[45:90], neg_val[45:90])
    count3 = Count(pos_val[90:135], neg_val[45:90])
    count += (count1 + count3 + count3) 
    print('Testing Accuracy: First : ' + '{:.09f}'.format(count1 / 45.0) + ' Second : ' + '{:.09f}'.format(count2 / 45.0) + ' Third : ' + '{:.09f}'.format(count3 / 45.0))
    print('Batch total Accuracy : ' + '{:.09f}'.format((count1 + count2 + count3)/ 135.0))
    return count     

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

    shape_sketch = tf.shape(sketch_dense)
    # Euclidean Distance
    dist_pos = EuclideanDist(sketch_dense, image_pos_dense)
    dist_neg = EuclideanDist(sketch_dense, image_neg_dense)
    margins = tf.constant(margin, dtype = tf.float32, shape = [batch_size])
    print(dist_pos, dist_neg, margins)

    with tf.name_scope('Loss') as scope:
        zeros = tf.constant(0.0, dtype = tf.float32, shape = [batch_size])
        regularizers = tf.nn.l2_loss(image_weights['wc1']) + tf.nn.l2_loss(image_weights['wc2']) + tf.nn.l2_loss(image_weights['wc3']) \
                        + tf.nn.l2_loss(image_weights['wc4']) + tf.nn.l2_loss(image_weights['wc5']) + tf.nn.l2_loss(image_weights['wd1']) \
                        + tf.nn.l2_loss(image_weights['wd2']) \
                        + tf.nn.l2_loss(sketch_weights['wc1']) + tf.nn.l2_loss(sketch_weights['wc2']) + tf.nn.l2_loss(sketch_weights['wc3']) \
                        + tf.nn.l2_loss(sketch_weights['wc4']) + tf.nn.l2_loss(sketch_weights['wc5']) + tf.nn.l2_loss(sketch_weights['wd1']) \
                        + tf.nn.l2_loss(sketch_weights['wd2'])
        cost = tf.reduce_sum( tf.nn.relu(margins + dist_pos - dist_neg) ) + beta * regularizers
        tf.summary.scalar("loss", cost)
    
    with tf.name_scope('Optimizer') as scope:
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, 200, 0.98, staircase = True, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost, global_step = global_step)
        tf.summary.scalar('global step', global_step)
        tf.summary.scalar('learning_rate', learning_rate)

    # Test correct order Accuray
    with tf.name_scope('Accuracy') as scope:
        less = tf.less(dist_pos, dist_neg, name ='Less')
        batch_count = tf.reduce_sum(tf.cast(less, tf.float32))
        batch_Accuracy = tf.divide(batch_count, 135.0)
        tf.summary.scalar('Accuracy', batch_Accuracy)

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
        
        # train
        step = 1
        train_data = ReadData(sess, batch_size, is_train = True)
        test_data = ReadData(sess, batch_size, is_train = False)
        while step * batch_size <= training_iters:
            s, ipos, ineg = next(train_data)

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
            
            if step % test_step == 0:               
                index = 1
                count = 0
                while index * batch_size <= 117*45:
                    s, ipos, ineg = next(test_data)
                    shape_s, b_count, b_Accuracy = sess.run([shape_sketch, batch_count, batch_Accuracy], feed_dict = {sketchs_placeholder : s, images_neg_placeholder : ipos, 
                                                images_pos_placeholder : ineg, keep_prob: 1.0})
                    print('Batch test: ', index)
                    print('Batch total Accuracy : ' + '{:.09f}'.format(b_Accuracy))
                    count += b_count
                    index += 1  
                accuracy = count / (117*45)
                print('Tensor Shape: ', shape_s)
                print('Total Accuracy : ', '{:.09f}'.format(accuracy)) 

            step += 1
        
        print("Optimization Finished!")
        

        # test 
        index = 1
        count = 0
        while index * batch_size <= 117*45:
            s, ipos, ineg = next(test_data)
            pos_val, neg_val = sess.run([dist_pos, dist_neg], feed_dict = {sketchs_placeholder : s, images_neg_placeholder : ipos, 
                                                images_pos_placeholder : ineg, keep_prob: 1.0})
            print('Batch test: ', index)
            tmp = Test(pos_val = pos_val, neg_val = neg_val)
            count += tmp
            index += 1  
        print('Total Accuracy : ', '{:.09f}'.format(count / (117*45)))

if __name__ == '__main__':
    run_training()
