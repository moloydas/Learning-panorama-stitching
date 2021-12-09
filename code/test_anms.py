import cv2
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from scipy import ndimage
from skimage.feature import peak_local_max

from anms import anms ,anms_vectorised
import time

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

if __name__ == '__main__':
    data_dir = '../Data/Train/Set1'
    img_list = os.listdir(data_dir)

    for img_name in img_list:

        img = cv2.imread(os.path.join(data_dir,img_name), 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detections, corner_fn = detect_harris_coreners(gray)
        print('points detected '+ str(detections.shape))
        img_w_corners = show_harris_corners(img, detections)

        # corners = detect_shi_tomasi_coreners(gray)
        # plt.imshow(corner_fn)
        # plt.show()

        # start_time = time.time()
        # anms_corners = anms(detections, 500, corner_fn)
        # print('time taken by old algo: ' + str(time.time() - start_time))
        # print(anms_corners.shape)
        # img_w_anms_corners = show_harris_corners(img, anms_corners)

        start_time = time.time()
        anms_corners_vect = anms_vectorised(detections, 200, corner_fn)
        print('time taken by new algo: ' + str(time.time() - start_time))
        print('final points after anms: '+ str(anms_corners_vect.shape))
        img_w_anms_corners_vec = show_harris_corners(img, anms_corners_vect)

        # cv2.imshow('img', img)
        # cv2.imshow('corner_fn', corner_fn)
        cv2.imshow('img_w_corners', img_w_corners)
        # cv2.imshow('img_w_anms_corners', img_w_anms_corners)
        cv2.imshow('img_w_anms_corners_vec', img_w_anms_corners_vec)
        k = cv2.waitKey(0)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()
