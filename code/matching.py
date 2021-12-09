# matching.py
import numpy as np
import cv2
import os
from feature_detection import *
from anms import *
from feature_descriptor import *

def show_matches(matches, img_1, features_1, img_2, features_2, mask=None):
    img = np.hstack((img_1, img_2))

    if mask is None:
        mask = np.ones((len(matches), 1))

    for idx, match in enumerate(matches):
        if mask[idx] == 0:
            color = (0,0,255)
        else:
            color = (0,255,0)

        pt1 = features_1[match[0], :]
        pt2 = features_2[match[1], :].copy()
        pt2[0] += img_1.shape[1]

        img = cv2.circle(img, tuple(pt1), 5, (255,0,0), 1)
        img = cv2.circle(img, tuple(pt2), 5, (255,0,0), 1)
        img = cv2.line(img, tuple(pt1), tuple(pt2), color, 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# converts img points from (row, col) to (x,y)
def reverse_pts(corners):
    new_corners = np.zeros_like(corners)
    for idx, pt in enumerate(corners):
        x,y = pt.ravel()
        new_corners[idx] = [y,x]
    return new_corners

#sum of squared differences
def ssd(vec_1, vec_2):
    return np.sum((vec_1 - vec_2)**2, axis=1)

def match(feature_1, feature_des_1, feature_2, feature_des_2):
    matches = []
    rejected = []
    for idx, feature_des in enumerate(feature_des_1):
        distances = ssd(feature_des, feature_des_2)
        dist_small_idx = np.argpartition(distances, 2)
        ratio = distances[dist_small_idx[0]]/distances[dist_small_idx[1]]

        if ratio < 0.7:
            matches.append([idx, dist_small_idx[0]])
        else:
            rejected.append([idx, dist_small_idx[0]])

    return matches, rejected

if __name__ == '__main__':
    data_dir = '../Data/Train/Set1'
    img_name_list = os.listdir(data_dir)

    img_name_list.sort()

    img_list = []
    feature_vec_list = []
    feature_des_list = []

    for img_name in img_name_list:
        print(f'Processing {img_name}')
        img = cv2.imread(os.path.join(data_dir,img_name), 1)
        print(f'img_size: {img.shape}')
        img_list.append(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #feature detections
        detections, corner_fn = detect_harris_coreners(gray)

        #anms
        anms_corners_vect, r = anms_vectorised(detections, 200, corner_fn)
        anms_corners_vect = reverse_pts(anms_corners_vect)
        feature_vec_list.append(anms_corners_vect)

        #description
        feature_des_vec = extract_descriptor(anms_corners_vect, gray, 8, 5)
        feature_des_list.append(feature_des_vec)

    #Match features between images
    for i in range(len(feature_vec_list)):

        matches, rejected = match(  feature_vec_list[i], 
                                    feature_des_list[i],
                                    feature_vec_list[int((i+1)%len(feature_vec_list))],
                                    feature_des_list[int((i+1)%len(feature_vec_list))])

        print(f'matches found: {len(matches)}')
        print(f'rejections: {len(rejected)}')

        show_matches(matches, 
                    img_list[i], 
                    feature_vec_list[i],
                    img_list[int((i+1)%len(feature_vec_list))], 
                    feature_vec_list[int((i+1)%len(feature_vec_list))])

