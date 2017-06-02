#!/usr/bin/env python
# -*- coding:utf-8 -*- 

import tensorflow as tf 
import numpy as np 
import os
import cv2
from PIL import Image

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

DEPTH = 3 

def convert_to(data_path, name):

    rows = 256
    cols = 256
    depth = DEPTH

    writer = tf.python_io.TFRecordWriter(name + '.tfrecords')

    for img_name in os.listdir(data_path):
        img_path = data_path + '\\' + img_name
        img = Image.open(img_path)
        img = img.resize((256, 256))
        img_raw = img.tobytes()

        label = img_name[:-4]
        label = int(label)
        example = tf.train.Example(features = tf.train.Features(feature = {
                                    'image_label': _int64_feature(label),
                                    'image_raw':_bytes_feature(img_raw)
        }))

        writer.write(example.SerializeToString())
    writer.close()

data_path = r'E:\work\Python\shoes\train\images'
name = 'shoes_images_train'

convert_to(data_path, name)

data_path = r'E:\work\Python\shoes\train\sketches'
name = 'shoes_sketches_train'

convert_to(data_path, name)