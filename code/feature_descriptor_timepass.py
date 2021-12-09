import numpy as np
import cv2
import os
from feature_detection import *
from anms import *

def extract_descriptor(features, img, kernel, stride):
    img_padded = img.copy()
    img_padded = cv2.copyMakeBorder(img, int(kernel*stride/2), int(kernel*stride/2), int(kernel*stride/2), int(kernel*stride/2), cv2.BORDER_CONSTANT)

    feature_vector_list = []

    for i in range(features.shape[0]):
        poi = features[i,:].astype(np.int32) + 20
        patch_size = kernel*stride
        patch = img_padded[poi[0]-int(patch_size/2):poi[0]+int(patch_size/2), poi[1]-int(patch_size/2):poi[1]+int(patch_size/2)]
        patch = cv2.GaussianBlur(patch, (stride,stride), 0)
        feature_matrix = cv2.resize(patch, (kernel, kernel), interpolation=cv2.INTER_NEAREST)
        feature_vector = feature_matrix.flatten().astype(np.float32)
        feature_vector -= np.mean(feature_vector)
        feature_vector /= np.std(feature_vector)
        feature_vector_list.append(feature_vector)

    return np.array(feature_vector_list)

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

        key_pts = []
        for i in range(anms_corners_vect.shape[0]):
            pt = cv2.KeyPoint(anms_corners_vect[i,1],anms_corners_vect[i,0], corner_fn[int(anms_corners_vect[i,0]), int(anms_corners_vect[i,1])])
            key_pts.append(pt)
        feature_key_pt_list.append(key_pts)

        #description
        feature_vector = extract_descriptor(anms_corners_vect, gray, 8, 5)
        feature_des_list.append(feature_vector)

    # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    # knn_matches = matcher.knnMatch(feature_des_list[0], feature_des_list[0], 2)

    # #Filter matches using the Lowe's ratio test
    # ratio_thresh = 0.7
    # good_matches = []
    # for m,n in knn_matches:
    #     if m.distance < ratio_thresh * n.distance:
    #         good_matches.append(m)

    # #Draw matches
    # img_matches = np.empty((max(img_list[0].shape[0], img_list[1].shape[0]), img_list[0].shape[1]+img_list[1].shape[1], 3), dtype=np.uint8)
    # cv2.drawMatches(img_list[0], feature_key_pt_list[0], img_list[1], feature_key_pt_list[1], good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # #show matches
    # cv2.imshow('Good Matches', img_matches)
    # cv2.waitKey(0)

    # #     cv2.imshow('img_w_anms_corners_vec', img_w_anms_corners_vec)
    # #     k = cv2.waitKey(0)
    # #     if k == ord('q'):
    # #         cv2.destroyAllWindows()
    # #         break

    # # cv2.destroyAllWindows()

