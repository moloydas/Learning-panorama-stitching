import numpy as np
import cv2
import os
from feature_detection import *
from anms import *
import math

def extract_descriptor(features, img, kernel, stride):
    img_copy = img.copy()
    patch_size = kernel*stride
    padding = int(patch_size/2)
    img_copy = cv2.copyMakeBorder(img_copy, padding, padding, padding, padding, cv2.BORDER_CONSTANT)

    feature_vec_list = []

    for i in range(features.shape[0]):
        x, y = (features[i].astype(int) + padding) - int(patch_size/2) + 1
        patch = img_copy[y:y+patch_size, x:x+patch_size]
        blur_patch = cv2.GaussianBlur(patch, (stride, stride), 0)
        sub_blur_patch = cv2.resize(blur_patch, (kernel, kernel), cv2.INTER_AREA)
        sub_blur_patch = sub_blur_patch.reshape(-1)
        feature_vec = (sub_blur_patch - np.average(sub_blur_patch))/ np.std(sub_blur_patch)
        feature_vec_list.append(feature_vec)

    return np.array(feature_vec_list)

if __name__ == '__main__':
    data_dir = '../Data/Train/Set1'
    img_name_list = os.listdir(data_dir)

    img_name_list.sort()

    img_list = []
    feature_key_pt_list = []
    feature_des_list = []

    for img_name in img_name_list:
        img = cv2.imread(os.path.join(data_dir,img_name), 1)
        img_list.append(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #feature detections
        detections, corner_fn = detect_harris_coreners(gray)

        #anms
        anms_corners_vect, r = anms_vectorised(detections, 200, corner_fn)

        #description
        feature_des_vec = extract_descriptor(anms_corners_vect, gray, 8, 5)

        break

