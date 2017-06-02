#!/usr/bin/env python
#-*- coding:utf-8 -*-

import tensorflow as tf 
import numpy as np 
import os 
import json 

def read_and_decode(filename):
    print(filename)
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features = {
                                        'image_label': tf.FixedLenFeature([], tf.int64),
                                        'image_raw': tf.FixedLenFeature([], tf.string)
    })
    label = tf.cast(features['image_label'], tf.int32)
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    print(label ,img)
    return label, img


test_images_name = []
test_sketchs_name = []
test_triplets = []

train_images_name = []
train_images_sketch = []
train_triplets = []

def LoadTriplets(filename):
    with open(filename, encoding='utf-8') as f:
        info = json.load(f)
        test_images_name = info['test']['images']
        test_sketchs_name = info['test']['sketches']
        test_triplets = info['test']['triplets']

        train_images_name = info['train']['images']
        train_sketchs_name = info['train']['sketches']
        train_triplets = info['train']['triplets']

    #print(len(train_triplets))
    #print(train_images_name[0])
    #print(train_sketchs_name[0])
    #print(train_triplets[0][0])
    return (test_images_name, test_sketchs_name, test_triplets, train_images_name, train_sketchs_name, train_triplets)


def ReadData(sess, batch_size = 128):
    s = None
    ipos = None
    ineg = None

    shoes_annotation = r'../shoes_annotation.json'

    print(r'Loading ' + shoes_annotation + r'...')
    test_images_name, test_sketchs_name, test_triplets, train_images_name, train_sketchs_name, train_triplets = LoadTriplets(shoes_annotation)

    
    print(len(train_triplets))

    filename1 = r'./shoes_images_train.tfrecords'
    filename2 = r'./shoes_sketches_train.tfrecords'
    print(r'Loading ' + filename1 + r'...')
    label_images, shoes_images_train = read_and_decode(filename1)
    print(r'Loading ' + filename2 + r'...')
    label_sketches, shoes_sketches_train = read_and_decode(filename2)
    
    shoes_images = []
    for i in range(len(train_images_name)):
        l, shoes_image = sess.run([label_images, shoes_images_train])
        shoes_images.append(shoes_image)
    
    print(len(shoes_images))

    shoes_sketchs = []
    for i in range(len(train_sketchs_name)):
        l, shoes_sketch = sess.run([label_sketches, shoes_sketches_train])
        shoes_sketchs.append(shoes_sketch)
    print(len(shoes_sketchs))
    
    print(shoes_images_train.get_shape())

    for i in range(304 * 45):
        t0 = int(i / 45)
        t1 = i % 45

        t = train_triplets[t0][t1]
        t2 = t[0]
        t3 = t[1]

        if s is None:
            s = shoes_sketchs[t0]
        else:
            s = tf.concat(0, [s, shoes_sketchs[t0]])
        
        if ipos is None:
            ipos = shoes_images[t2]
        else:
            ipos = tf.concat(0, [ipos, shoes_images[t2]])

        if ineg is None:
            ineg = shoes_images[t3]
        else:
            ineg = tf.concat(0, [ineg, shoes_images[t3]])

        if i % batch_size == 0:
            yield s, ipos, ineg
            s = None
            ipos = None
            ineg = None



if __name__ == '__main__':
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        a = ReadData(sess)
        next(a)
        next(a)
