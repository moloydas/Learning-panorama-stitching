#feature_detection.py
import cv2
import os
import numpy as np
from skimage.feature import peak_local_max

def detect_harris_coreners(gray_img, max_pnts=2000):
    gray_img = np.float32(gray_img)
    corner_fn = cv2.cornerHarris(gray_img, 3, 3, 0.04)
    corners = peak_local_max(corner_fn, min_distance=7, num_peaks=max_pnts, threshold_abs= 0.000001)

    return corners, corner_fn

def show_harris_corners(img, corners):
    img_w_corners = img.copy()
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img_w_corners, (y, x), 3, (0, 0, 255), -1)
    return img_w_corners

def detect_shi_tomasi_coreners(gray_img):
    corners = cv2.goodFeaturesToTrack(gray, 500, 0.01, 10)
    corners = np.int0(corners)
    return corners

def show_shi_tomsai_corners(img, corners):
    img_w_corners = img.copy()
    for i in corners:
        x,y = i.ravel()     
        cv2.circle(img_w_corners, (x,y), 3, (0, 0, 255), -1)
    return img_w_corners
