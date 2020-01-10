#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 12:46:54 2019

@author: avinashshanker
ID: 1001668570
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def sobel(image, filter, convert_to_degree=False):
    inputx_image = convolve(image, filter)
    plt.imshow(inputx_image, cmap='gray')
    plt.title("Edge in Y axis")
    plt.show()

    inputy_image = convolve(image, np.flip(filter.T, axis=0))
    plt.imshow(inputy_image, cmap='gray')
    plt.title("Edge in X axis")
    plt.show()

    gradient_magnitude = np.sqrt(np.square(inputx_image) + np.square(inputy_image))

    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title("Magnitude")
    plt.show()

    orientation = np.arctan2(inputy_image, inputx_image)

    if convert_to_degree:
        orientation = np.rad2deg(orientation)
        orientation += 180

    return gradient_magnitude, orientation

def dnorm(x, mu, sd):
    value = 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)
    return value


def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max()
    return kernel_2D


def gaussian_blur(image, kernel_size, verbose=False):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))
    return convolve(image, kernel, average=True)

def convolve(image, kernel, average=False):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    output = np.zeros(image.shape)
    height = int((kernel_row - 1) / 2)
    width = int((kernel_col - 1) / 2)
    mod_image = np.zeros((image_row + (2 * height), image_col + (2 * width)))
    mod_image[height:mod_image.shape[0] - height, width:mod_image.shape[1] - width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * mod_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
    return output

def non_max_suppression(gradient_magnitude, orientation):
    image_row, image_col = gradient_magnitude.shape
    output = np.zeros(gradient_magnitude.shape)
    PI = 180
    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = orientation[row, col]
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]

            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]

            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]
    return output


def threshold(image, low, high, weak):
    output = np.zeros(image.shape)
    strong = 255
    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))
    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak
    return output


def Boundry_Detection(image, weak):
    image_row, image_col = image.shape
    top_to_bottom = image.copy()
    for row in range(1, image_row):
        for col in range(1, image_col):
            if top_to_bottom[row, col] == weak:
                if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[row - 1, col] == 255 or top_to_bottom[
                    row + 1, col] == 255 or top_to_bottom[
                    row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[row - 1, col + 1] == 255 or top_to_bottom[
                    row + 1, col + 1] == 255:
                    top_to_bottom[row, col] = 255
                else:
                    top_to_bottom[row, col] = 0

    bottom_to_top = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(image_col - 1, 0, -1):
            if bottom_to_top[row, col] == weak:
                if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[row - 1, col] == 255 or bottom_to_top[
                    row + 1, col] == 255 or bottom_to_top[
                    row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[row - 1, col + 1] == 255 or bottom_to_top[
                    row + 1, col + 1] == 255:
                    bottom_to_top[row, col] = 255
                else:
                    bottom_to_top[row, col] = 0

    right_to_left = image.copy()

    for row in range(1, image_row):
        for col in range(image_col - 1, 0, -1):
            if right_to_left[row, col] == weak:
                if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[row - 1, col] == 255 or right_to_left[
                    row + 1, col] == 255 or right_to_left[
                    row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[row - 1, col + 1] == 255 or right_to_left[
                    row + 1, col + 1] == 255:
                    right_to_left[row, col] = 255
                else:
                    right_to_left[row, col] = 0

    left_to_right = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(1, image_col):
            if left_to_right[row, col] == weak:
                if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[row - 1, col] == 255 or left_to_right[
                    row + 1, col] == 255 or left_to_right[
                    row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[row - 1, col + 1] == 255 or left_to_right[
                    row + 1, col + 1] == 255:
                    left_to_right[row, col] = 255
                else:
                    left_to_right[row, col] = 0

    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right

    final_image[final_image > 255] = 255

    return final_image


if __name__ == '__main__':
    Image_Name = input("\nPlease Enter the Image name Eg: 1.png >>")
    image = cv2.imread(Image_Name)
    blurred_image = gaussian_blur(image, kernel_size=9)
    edge_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient_magnitude, orientation = sobel(blurred_image, edge_filter, convert_to_degree=True)
    processed_img = non_max_suppression(gradient_magnitude, orientation)
    processed_img = threshold(processed_img, 1, 20, weak=1)
    processed_img = Boundry_Detection(processed_img, weak=1)
    cv2.imwrite('prob1-1.png',processed_img)
    Final_output = cv2.imread('prob1-1.png')
    plt.imshow(Final_output, cmap='gray')
    plt.title("Image Boundry Trace")
    plt.show()


