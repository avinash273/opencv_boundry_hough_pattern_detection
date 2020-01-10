#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 12:46:54 2019

@author: avinashshanker
ID: 1001668570
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL.Image import *
from scipy import ndimage

def Harris(input_image,sigma):
    Image_Frame = image_mapper(input_image)
    image_array = np.asarray(Image_Frame, dtype=np.float64)
    gx = ndimage.sobel(image_array, 0)
    gy = ndimage.sobel(image_array, 1)
    gx_2 = gx * gx
    gy_2 = gy * gy
    ixy = gx * gy
    gx_2 = ndimage.gaussian_filter(gx_2, sigma)
    gy_2 = ndimage.gaussian_filter(gy_2, sigma)
    ixy = ndimage.gaussian_filter(ixy, sigma)
    c, l = image_array.shape
    result = np.zeros((c, l))
    r = np.zeros((c, l))
    rmax = 0
    for x in range(c):
        for j in range(l):
            m = np.array([[gx_2[x, j], ixy[x, j]], [ixy[x, j], gy_2[x, j]]], dtype=np.float64)
            r[x, j] = np.linalg.det(m) - 0.04 * (np.power(np.trace(m), 2))
            if r[x, j] > rmax:
                rmax = r[x, j]
    for x in range(c - 1):
        for j in range(l - 1):
            if r[x, j] > 0.01 * rmax and r[x, j] > r[x-1, j-1] and r[x, j] > r[x-1, j+1]\
                                     and r[x, j] > r[x+1, j-1] and r[x, j] > r[x+1, j+1]:
                result[x, j] = 1

    pc, pr = np.where(result == 1)
    return input_image,pc,pr

def image_mapper(input_image):
    a = Image.getpixel(input_image, (0, 0))
    if type(a) == int:
        return input_image
    else:
        c, l = input_image.size
        image_array = np.asarray(input_image)
        Image_Frame = np.zeros((l, c))
        for x in range(l):
            for j in range(c):
                t = image_array[x, j]
                ts = sum(t)/len(t)
                Image_Frame[x, j] = ts
        return Image_Frame

def main():
    
    Image_Name = input("\nPlease Enter the Image name Eg: 2-2.jpg >>")
    input1 = open(Image_Name)
    input2 = open('2-1.jpg')
    out_image1, pc1, pr1 = Harris(input1,1)
    out_image12, pc12, pr12 = Harris(input1,3)
    out_image13, pc13, pr13 = Harris(input1,5)
    out_image2, pc2, pr2 = Harris(input2,1)
    
    ##############################################
    #This part of code only for Displaying  images
    ##############################################
    fig1 = plt.figure(figsize = (15,15))
    a = fig1.add_subplot(3, 2, 1)
    fig1.suptitle('Problem 2', fontsize=16)
    plt.imshow(input1,cmap='gray')
    a.set_title('Cow Image Original')
    
    a = fig1.add_subplot(3, 2, 2)
    plt.plot(pr1, pc1, 'r+')
    plt.imshow(out_image1,cmap='gray')
    a.set_title('Corner Sigma=1')
    
    a = fig1.add_subplot(3, 2, 3)
    fig1.suptitle('Problem  2', fontsize=16)
    plt.plot(pr12, pc12, 'r+')
    plt.imshow(out_image12,cmap='gray')
    a.set_title('Corner Sigma=3')
    
    a = fig1.add_subplot(3, 2, 4)
    plt.plot(pr13, pc13, 'r+')
    plt.imshow(out_image13,cmap='gray')
    a.set_title('Corner Sigma=5')
    
    fig1 = plt.figure(figsize = (12,12))
    a = fig1.add_subplot(2, 1, 1)
    fig1.suptitle('Problem 2', fontsize=16)
    plt.imshow(input2,cmap='gray')
    a.set_title('Chess Board Original')
    
    a = fig1.add_subplot(2, 1, 2)
    plt.plot(pr2, pc2, 'r+')
    plt.imshow(out_image2,cmap='gray')
    a.set_title('Chess Board Corner')

if __name__ == "__main__":
    main()






#https://stackoverflow.com/questions/43525409/harris-corner-detector-python