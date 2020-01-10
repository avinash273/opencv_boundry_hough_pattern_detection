#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 12:46:54 2019

@author: avinashshanker
ID: 1001668570
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage.filters import sobel
from collections import defaultdict
import scipy.io as sio

def accumulator(Dictionarys, grayImage):
    grad = gradient_orientation(grayImage)
    accumulator = np.zeros(grayImage.shape)
    for (i,j),value in np.ndenumerate(grayImage):
        if value:
            for r in Dictionarys[grad[i,j]]:
                accum_i, accum_j = i+r[0], j+r[1]
                if accum_i < accumulator.shape[0] and accum_j < accumulator.shape[1]:
                    accumulator[accum_i, accum_j] += 1
    return accumulator

def HoughTransform(edge_img, c, ptlist):
    c = (100,100)
    Dictionarys = create_table(edge_img, c)
    def f(query_image):
        return accumulator(Dictionarys, query_image)       
    return f 

def gradient_orientation(image):
    gx = sobel(image, axis=0, mode='constant')
    gy = sobel(image, axis=1, mode='constant')
    grad = np.arctan2(gy,gx) * 180 / np.pi
    return grad

def create_table(image, origin):
    grad = gradient_orientation(image)
    Dictionarys = defaultdict(list)
    for (i,j),value in np.ndenumerate(image):
        if value:
            Dictionarys[grad[i,j]].append((origin[0]-i, origin[1]-j))
    return Dictionarys

def n_max(a, n):
    indices = a.ravel().argsort()[-n:]
    indices = (np.unravel_index(i, a.shape) for i in indices)
    return [(a[i], i) for i in indices]

def print_output(gh, reference_image, query):
    accumulator = gh(query)
    
    fig = plt.figure(figsize = (12,12))    
    fig.add_subplot(2,2,1)
    plt.axis('off')
    plt.title('Original')
    plt.imshow(reference_image, cmap='gray')
    
    fig.add_subplot(2,2,2)
    plt.title('Detect Image')
    plt.axis('off')
    plt.imshow(query, cmap='gray')
    
    fig.add_subplot(2,2,3)
    plt.title('Accumulator Output')
    plt.axis('off')
    plt.imshow(accumulator, cmap='gray')
    
    fig.add_subplot(2,2,4)
    plt.axis('off')
    plt.title('Detection for Accumulator=2 ')
    plt.imshow(query, cmap='gray')
    
    m = n_max(accumulator, 2)
    y_points = [pt[1][0] for pt in m]
    x_points = [pt[1][1] for pt in m]
    print("\nFirst Circle coordinates: (%d,%d)"%(x_points[0],y_points[0]))
    print("\nSecond Circle coordinates: (%d,%d)"%(x_points[1],y_points[1]))
    plt.scatter(x_points, y_points, marker='+', color= 'r')
    return


def main():
    Image_mat = input("\nPlease Enter the Image name Eg: train.mat >>")
    i=sio.loadmat(Image_mat)
    i.keys()
    c = i['c']
    ptlist=i['ptlist']
       
    val =[]
    for row in ptlist:
        for i in row:
            a = i[0]
            val.append(a)
    ptlist = np.array(val)
    
    Image_train = input("\nPlease Enter the Image name Eg: train.png >>")
    Image_test = input("\nPlease Enter the Image name Eg: test.png >>")
    input_image = cv2.imread(Image_train)
    test_img = cv2.imread(Image_test)
    image_tranform = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    grayscaler = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    edge_img = cv2.Canny(grayscaler, threshold1 = 0, threshold2 = 50, apertureSize = 3)
    a = cv2.Canny(image_tranform,threshold1 = 0, threshold2 = 50, apertureSize = 3)
    print_output(HoughTransform(edge_img, c, ptlist), input_image, a)
    
    test_img = cv2.imread("test2.png")
    image_tranform = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    grayscaler = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    edge_img = cv2.Canny(grayscaler, threshold1 = 0, threshold2 = 50, apertureSize = 3)
    a = cv2.Canny(image_tranform,threshold1 = 0, threshold2 = 50, apertureSize = 3)
    print_output(HoughTransform(edge_img, c, ptlist), input_image, a)

if __name__ == "__main__":
    main()
