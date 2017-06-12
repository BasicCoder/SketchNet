#!/usr/bin/env python
#-*- coding:utf-8 -*-

import cv2
import numpy as np 
import os 
import json

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

def ProcessImagesData():
    shoes_annotation = r'./Data/sbir_cvpr2016/shoes/annotation/shoes_annotation.json'
    print(r'Loading ' + shoes_annotation + r'...')
    test_images_name, test_sketchs_name, test_triplets, train_images_name, train_sketchs_name, train_triplets = LoadTriplets(shoes_annotation)

    img_path = r'./Data/sbir_cvpr2016/shoes/test/images/'
    write_img_path = r'./Data/sbir_cvpr2016/shoes/test/canny_images/'
    for img in test_images_name:
        print('Process ' + img + '...')
        image = cv2.imread(img_path + img)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        canny = cv2.Canny(image, 50, 150)
        canny_sketch = cv2.bitwise_not(canny)
        canny_sketch = cv2.cvtColor(canny_sketch,  cv2.COLOR_GRAY2RGB)
        print('Write ' + img + '...')
        #cv2.imshow(img, canny_sketch)
        #cv2.waitKey(0)
        cv2.imwrite(write_img_path + img, canny_sketch, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

'''
img_path = r'./Data/sbir_cvpr2016/shoes/train/images/12.jpg'
skt_path = r'./Data/sbir_cvpr2016/shoes/train/sketches/12.png'
img = cv2.imread(img_path)
skt = cv2.imread(skt_path)



img = cv2.GaussianBlur(img, (3, 3), 0)
canny = cv2.Canny(img, 50, 150)
canny_sketch = cv2.bitwise_not(canny)
canny_sketch = cv2.cvtColor(canny_sketch,  cv2.COLOR_GRAY2RGB)
cv2.imwrite('./12.jpg', canny_sketch, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

cv2.imshow('Image', img)
cv2.imshow('Sketch', skt)
cv2.imshow('Canny', canny_sketch)
cv2.waitKey(0)
'''
if __name__ == '__main__':
    ProcessImagesData()