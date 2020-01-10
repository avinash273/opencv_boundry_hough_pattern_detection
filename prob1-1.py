#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:02:46 2019

@author: avinashshanker
ID: 1001668570
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def myGaussianSmoothing(I, k, s):
    ax = np.linspace(-(k - 1) / 2., (k - 1) / 2., k)
    x, y = np.meshgrid(ax, ax)
    exp = np.exp(-0.5 * (np.square(x) + np.square(y)) / np.square(s))
    divide = 2*np.pi*s*s
    kernel = exp/divide
    img_row, img_col = I.shape[0],I.shape[1]
    kernel_row,kernel_col = kernel.shape[0],kernel.shape[1]
    output = np.zeros(I.shape)
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((img_row + (2 * pad_height), img_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = I
    for row in range(img_row):
        for col in range(img_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
    return output

def gradient(I):
    gx,gy = np.gradient(I)
    gx2 = gx*gx
    gy2 = gy*gy
    magnitude = np.sqrt(gx2 + gy2)
    orientation = np.arctan(gy//gx)
    return magnitude,orientation 

def quiverI():
	I = Image.open('1_gray_smooth_ori.png')
	x_axis, y = np.mgrid[0:I.size[0]:500j, 0:I.size[1]:500j]
	gx, gy = np.gradient(I)
	skip = (slice(None, None, 6), slice(None, None, 6))
	fig, ax = plt.subplots(figsize = (12,12))
	ax.imshow(I,extent=[x_axis.min(), x_axis.max(), y.min(), y.max()],cmap='gray')
	ax.quiver(x_axis[skip], y[skip], gx[skip], gy[skip], color = 'r')
	ax.set(aspect=1, title='Quiver Plot')
	plt.show()
    


def main():
    ## Please change input image here
    Image_Name = input("\nPlease Enter the Image name Eg: 1.png >>")
    image_gray = cv2.imread(Image_Name,0)
    val = 3.0
    val = input("\nEnter Gauss Smooth Coeff Eg: 3 >>")
    val = float(val)
    I_smooth = myGaussianSmoothing(image_gray,7, val)
    magnitude,orientation = gradient(I_smooth)
    cv2.imwrite('1_gray_smooth_ori.png',magnitude)
    [gx,gy] = np.gradient(I_smooth)
    
    
    ##############################################
    #This part of code only for Displaying  images
    ##############################################
    fig1 = plt.figure(figsize = (15,15))
    fig1.suptitle('Problem  1-1', fontsize=16)
    a = fig1.add_subplot(2, 2, 1)
    plt.imshow(image_gray,cmap='gray')
    a.set_title('Original')
    
    a = fig1.add_subplot(2, 2, 2)
    plt.imshow(I_smooth,cmap='gray')
    a.set_title('Gauss  Smooth')
    
    a = fig1.add_subplot(2, 2, 3)
    plt.imshow(magnitude,cmap='gray')
    a.set_title('magnitude')
    
    a = fig1.add_subplot(2, 2, 4)
    plt.imshow(orientation,cmap='gray')
    a.set_title('orientation')    
    quiverI()
    

if __name__ == "__main__":
    main()


