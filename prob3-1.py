#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 12:46:54 2019

@author: avinashshanker
ID: 1001668570
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def myHoughLine(img_bin, n): 
  nR,nC = img_bin.shape
  theta_res=1
  rho_res=1
  theta = np.linspace(-90.0, 0.0, np.ceil(90.0/theta_res) + 1.0)
  theta = np.concatenate((theta, -theta[len(theta)-2::-1]))
  D = np.sqrt((nR - 1)**2 + (nC - 1)**2)
  q = np.ceil(D/rho_res)
  nrho = 2*q + 1
  rho = np.linspace(-q*rho_res, q*rho_res, nrho)
  H = np.zeros((len(rho), len(theta)))
  for rowIdx in range(nR):
    for colIdx in range(nC):
      if img_bin[rowIdx, colIdx]:
        for thIdx in range(len(theta)):
          rhoVal = colIdx*np.cos(theta[thIdx]*np.pi/180.0) + \
              rowIdx*np.sin(theta[thIdx]*np.pi/180)
          rhoIdx = np.nonzero(np.abs(rho-rhoVal) == np.min(np.abs(rho-rhoVal)))[0]
          H[rhoIdx[0], thIdx] += 1
  return rho, theta, H, n

def accumulator(ht_acc_matrix, n, rhos, thetas):
  flat = list(set(np.hstack(ht_acc_matrix)))
  flat_sorted = sorted(flat, key = lambda n: -n)
  coords_sorted = [(np.argwhere(ht_acc_matrix == acc_value)) for acc_value in flat_sorted[0:n]]
  rho_theta = []
  x_y = []
  for coords_for_val_idx in range(0, len(coords_sorted), 1):
    coords_for_val = coords_sorted[coords_for_val_idx]
    for i in range(0, len(coords_for_val), 1):
      n,m = coords_for_val[i]
      rho = rhos[n]
      theta = thetas[m]
      rho_theta.append([rho, theta])
      x_y.append([m, n])
  return [rho_theta[0:n], x_y]

def valid_point(pt, ymax, xmax):
  x, y = pt
  if x <= xmax and x >= 0 and y <= ymax and y >= 0:
    return True
  else:
    return False

def round_tup(tup):
  x,y = [int(round(num)) for num in tup]
  return (x,y)

def set_accumulator(target_im, pairs):
  im_y_max, im_x_max, channels = np.shape(target_im)
  for i in range(0, len(pairs), 1):
    point = pairs[i]
    rho = point[0]
    theta = point[1] * np.pi / 180
    m = -np.cos(theta) / np.sin(theta)
    b = rho / np.sin(theta)
    left = (0, b)
    right = (im_x_max, im_x_max * m + b)
    top = (-b / m, 0)
    bottom = ((im_y_max - b) / m, im_y_max)
    pts = [pt for pt in [left, right, top, bottom] if valid_point(pt, im_y_max, im_x_max)]
    if len(pts) == 2:
      cv2.line(target_im, round_tup(pts[0]), round_tup(pts[1]), (255,0,255), 1)

def main():
    Image_Name = input("\nPlease Enter the Image name Eg: 3.png >>")
    img_orig = cv2.imread(Image_Name)
    img = img_orig[:,:,::-1]
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(bw, threshold1 = 25, threshold2 = 50, apertureSize = 3)
    rhos, thetas, H, n = myHoughLine(edges, 5)
    rho_theta_pairs, x_y_pairs = accumulator(H, n, rhos, thetas)
    im_w_lines = img.copy()
    set_accumulator(im_w_lines, rho_theta_pairs)
    
    edges = cv2.Canny(bw, threshold1 = 20, threshold2 = 50, apertureSize = 3)
    rhos, thetas, H, n = myHoughLine(edges, 7)
    rho_theta_pairs, x_y_pairs = accumulator(H, n, rhos, thetas)
    im_w_lines8 = img.copy()
    set_accumulator(im_w_lines8, rho_theta_pairs)
    
    ##############################################
    #This part of code only for Displaying  images
    ##############################################
    fig1 = plt.figure(figsize = (15,15))
    a = fig1.add_subplot(1, 2, 1)
    fig1.suptitle('Problem  3-1', fontsize=16)
    plt.imshow(img,cmap='gray')
    a.set_title('Gray Scale Image')
    
    a = fig1.add_subplot(1, 2, 2)
    plt.imshow(edges,cmap='gray')
    a.set_title('Image Edges')
    
    fig1 = plt.figure(figsize = (15,15))
    a = fig1.add_subplot(1, 2, 1)
    plt.imshow(im_w_lines,cmap='gray')
    a.set_title('For Lines n = 5')
    
    a = fig1.add_subplot(1, 2, 2)
    plt.imshow(im_w_lines8,cmap='gray')
    a.set_title('For Lines n = 7')
if __name__ == "__main__":
    main()





