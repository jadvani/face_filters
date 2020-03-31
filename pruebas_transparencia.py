# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 00:38:47 2017

@author: Javier
"""
import cv2
import numpy as np

def crea_alpha(img):
    b_channel, g_channel, r_channel = cv2.split(img)

    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 #creating a dummy alpha channel image.

    img_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_RGBA


def superponer(img1,img2):
    img1=crea_alpha(img1)
    b_channel, g_channel, r_channel = cv2.split(img2)
    img2=cv2.merge((b_channel, g_channel, r_channel, 0))
    h, w, depth = img2.shape

    result = np.zeros((h, w, 3), np.uint8)

    for i in range(h):
        for j in range(w):
            color1 = img1[i, j]
            color2 = img2[i, j]
            alpha = color2[3] / 255.0
            new_color = [ (1 - alpha) * color1[0] + alpha * color2[0],(1 - alpha) * color1[1] + alpha * color2[1],(1 - alpha) * color1[2] + alpha * color2[2] ]
            result[i, j] = new_color
            return result
                  