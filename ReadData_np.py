#!/usr/bin/env python
#-*- coding:utf-8 -*-

#import matplotlib.pyplot as plt
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
    img = tf.image.convert_image_dtype( tf.image.decode_png( tf.read_file(image_path), channels = 3), dtype=tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    return img

def read_images(sess, data_path, images_name, is_train):
    images = []
    print('Loading ' +('train' if is_train else 'test') + ' images ...')
    for img in images_name:
        a = get_image(data_path + '/' + img)
        a = a.eval(session = sess)
        images.append(a)
    return images

def read_sketchs(sess, data_path, sketchs_name, is_train):
    sketchs = []
    print('Loading ' +('train' if is_train else 'test') + ' sketches ...')
    for img in sketchs_name:
        a = get_sketch(data_path + '/' + img)
        a = a.eval(session = sess)
        sketchs.append(a)
    return sketchs

def ReadData(sess, batch_size = 128, is_train = True):

    shoes_annotation = r'./Data/sbir_cvpr2016/shoes/annotation/shoes_annotation.json'
    print(r'Loading ' + shoes_annotation + r'...')
    test_images_name, test_sketchs_name, test_triplets, train_images_name, train_sketchs_name, train_triplets = LoadTriplets(shoes_annotation)
   
    if is_train:
        # Read tain image
        data_path = r'./Data/sbir_cvpr2016/shoes/train/images'
        shoes_images = read_images(sess, data_path, train_images_name, is_train)
        print(len(shoes_images))
        '''
        print('image :')
        plt.figure(1)
        plt.imshow(shoes_images[0])
        plt.show()
        plt.close(1)
        '''

        # Read train sketchs
        data_path = r'./Data/sbir_cvpr2016/shoes/train/sketches'
        shoes_sketchs = read_sketchs(sess, data_path, train_sketchs_name, is_train)
        print(len(shoes_sketchs))
        '''
        print('sketchs :')
        plt.figure(2)
        plt.imshow(shoes_sketchs[0])
        plt.show()
        plt.close(2)
        '''
    else:
        # Read tain image
        data_path = r'./Data/sbir_cvpr2016/shoes/test/images'
        shoes_images = read_images(sess, data_path, test_images_name, is_train)
        print(len(shoes_images))

        # Read train sketchs
        data_path = r'./Data/sbir_cvpr2016/shoes/test/sketches'
        shoes_sketchs = read_sketchs(sess, data_path, test_sketchs_name, is_train)
        print(len(shoes_sketchs))


    s = []
    ipos = []
    ineg = []

    if is_train:
        i = 0
        while True:
            sk_i = i   
            for j in range(len(train_triplets[i])):
                sk_i = i
                im_pos_i = train_triplets[sk_i][j][0]
                im_neg_i = train_triplets[sk_i][j][1]
                
                s.append(shoes_sketchs[sk_i])
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
            
            i += 1
            if(i >= len(train_triplets)):
                i = 0
    else:
        i = 0
        while True:
            sk_i = i
            for j in range(len(test_triplets[i])):
                sk_i = i
                im_pos_i = test_triplets[sk_i][j][0]
                im_neg_i = test_triplets[sk_i][j][0]

                s.append(shoes_sketchs[sk_i])
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
            
            i += 1
            if(i >= len(test_triplets)):
                i = 0
    # print(len(images_triplets))
    
    # tf.train.batch([images_triplets], batch_size = batch_size, num_threads=6)


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

        a = ReadData(sess, 5, True)
        next(a)
        next(a)