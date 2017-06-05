#!/usr/bin/env python
#-*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np 
import os 
import json 

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


def get_image(image_path):
    img = tf.image.convert_image_dtype( tf.image.decode_jpeg( tf.read_file(image_path), channels = 3),  dtype=tf.uint8 )
    img = tf.reshape(img, [256, 256, 3])
    return img

def get_sketch(image_path):
    img = tf.image.convert_image_dtype( tf.image.decode_png( tf.read_file(image_path), channels = 4), dtype=tf.uint8)
    img = tf.reshape(img, [256, 256, 4])
    return img

def ReadData(sess, batch_size = 128):

    shoes_annotation = r'./Data/sbir_cvpr2016/shoes/annotation/shoes_annotation.json'
    print(r'Loading ' + shoes_annotation + r'...')
    test_images_name, test_sketchs_name, test_triplets, train_images_name, train_sketchs_name, train_triplets = LoadTriplets(shoes_annotation)

    # Read all images
    shoes_images = []
    data_path = r'./Data/sbir_cvpr2016/shoes/train/images'
    print('Loading train images ...')
    for img in train_images_name:
        a = get_image(data_path + '/' + img)
        a = a.eval(session = sess)
        shoes_images.append(a)
    print(len(shoes_images))
    '''
    print('image :')
    plt.figure(1)
    plt.imshow(shoes_images[0])
    plt.show()
    plt.close(1)
    '''

    # Read all sketchs
    shoes_sketchs = []
    data_path = r'./Data/sbir_cvpr2016/shoes/train/sketches'
    print('Loading train sketches ...')
    for img in train_sketchs_name:
        b = get_sketch(data_path + '/' + img)
        b = b.eval(session = sess)
        shoes_sketchs.append(b)
    print(len(shoes_sketchs))
    '''
    print('sketchs :')
    plt.figure(2)
    plt.imshow(shoes_sketchs[0])
    plt.show()
    plt.close(2)
    '''
    s = []
    ipos = []
    ineg = []
    for i in range(len(train_triplets)):
        sk_i = i   
        for j in range(len(train_triplets[i])):
            s_i = i
            im_pos_i = train_triplets[sk_i][j][0]
            im_neg_i = train_triplets[sk_i][j][1]
            
            s.append(shoes_sketchs[s_i])
            ipos.append(shoes_images[im_pos_i])
            ineg.append(shoes_images[im_neg_i])
            length = len(s)
            if length != 0 and length % batch_size == 0:
                print(len(s), len(ipos), len(ineg))
                #print(s, ipos, ineg)
                yield s, ipos, ineg
                s = []
                ipos = []
                ineg = []

    # print(len(images_triplets))
    
    # tf.train.batch([images_triplets], batch_size = batch_size, num_threads=6)
    '''
    
    s = None
    ipos = None
    ineg = None
    # yield batch data
    for i in range(len(shoes_images) * 45):
        t0 = int(i / 45)
        t1 = i % 45

        t = train_triplets[t0][t1]
        t2 = t[0]
        t3 = t[1]
        print('t0 =', t0, 't1 =', t1, 't2 =', t2, 't3 =', t3)
        if s is None:
            s = tf.to_int32(shoes_sketchs[t0])
        else: 
            si = tf.to_int32(shoes_sketchs[t0])
            print(i, ':', s, si)
            s = tf.concat([s, si], 0)
        
        if ipos is None:
            ipos = tf.to_int32(shoes_images[t2])
        else:
            iposi = tf.to_int32(shoes_images[t2])
            #print(i, ':', ipos, iposi)
            ipos = tf.concat([ipos, iposi], 0)

        if ineg is None:
            ineg = tf.to_int32(shoes_images[t3])
        else:
            inegi = tf.to_int32(shoes_images[t3])
            ineg = tf.concat([ineg, inegi], 0)

        if i != 0 and i % (batch_size -1) == 0:
            print(s, ipos, ineg)
            yield s, ipos, ineg
            s = None
            ipos = None
            ineg = None
        '''

if __name__ == '__main__':
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        '''
        data_path = r'E:\work\Python\shoes\train\images\1.jpg'
        a = get_image(data_path)
        a = a.eval(session = sess)
        plt.figure(1)
        plt.imshow(a)
        plt.show()
        '''

        a = ReadData(sess, 5)
        next(a)
        next(a)