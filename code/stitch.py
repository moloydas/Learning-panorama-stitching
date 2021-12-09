# stitch.py

# ransac.py
import numpy as np
import cv2
import os
from feature_detection import *
from anms import *
from feature_descriptor import *
from matching import *
import time

# feature_1 = H(features_2)
# x' = H(x)
# x y 1 0 0 0 -x.x' -y.x'       x'
# 0 0 0 x y 1 -x.y' -y.y'       y'
def compute_homography(matches, feature_1, feature_2):
    # sample_range = len(matches)
    # sample_set = np.random.randint(sample_range, size=(8))

    A = []
    b = []

    print(f'number of match {len(matches)}')
    print(f'A expected size: {len(matches)*2} x 8')
    print(f'b expected size: {len(matches)*2} x 1')

    for match in matches:
        x, y = feature_2[match[1], :]
        x_, y_ = feature_1[match[0], :]
        A.append([x, y, 1, 0, 0, 0, -x*x_, -y*x_])
        A.append([0, 0, 0, x, y, 1, -x*y_, -y*y_])
        b.append(x_)
        b.append(y_)

    A = np.array(A)
    b = np.array(b).reshape(-1, 1)

    print(A.shape)
    print(b.shape)

    H = np.linalg.inv(A.T @ A) @ (A.T @ b)

    return H

# feature_1 = H(feature_2)
def compute_homography_opencv(matches, feature_1, feature_2):
    matches = np.array(matches)

    matched_features_1 = feature_1[matches[:,0]]
    matched_features_2 = feature_2[matches[:,1]]

    h, mask = cv2.findHomography(matched_features_2, matched_features_1, method=cv2.RANSAC, ransacReprojThreshold=5)

    return h, mask

if __name__ == '__main__':
    data_dir = '../Data/Train/Set1'
    img_name_list = os.listdir(data_dir)

    img_name_list.sort()

    img_list = []
    feature_vec_list = []
    feature_des_list = []

    for img_name in img_name_list:
        start_time = time.time()
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

        print(f'processing time: {time.time()-start_time}')

    print('Features extracted!!')
    print('Matching....')

    #Match features between images
    for i in range(len(feature_vec_list)):

        # if i != 2:
        #     continue

        matches, rejected = match(  feature_vec_list[i], 
                                    feature_des_list[i],
                                    feature_vec_list[int((i+1)%len(feature_vec_list))],
                                    feature_des_list[int((i+1)%len(feature_vec_list))])

        print(f'matches found: {len(matches)}')
        print(f'rejections: {len(rejected)}')

        print('Compute Homography!!')

        H, mask = compute_homography_opencv( matches, 
                                feature_vec_list[i], 
                                feature_vec_list[int((i+1)%len(feature_vec_list))])

        print(f'H:\n{H}')

        show_matches(matches,
                    img_list[i],
                    feature_vec_list[i],
                    img_list[int((i+1)%len(feature_vec_list))],
                    feature_vec_list[int((i+1)%len(feature_vec_list))],
                    mask)

        #####################################################
        ### Try 2: Deciding ref Image based on positive warped corners
        #####################################################
        row = img_list[i].shape[0]
        cols = img_list[i].shape[1]

        corners = np.array( [   [0, cols], 
                                [0, row],
                                [1, 1   ]])

        trans_corner = H @ corners
        trans_corner = trans_corner/trans_corner[2]

        x_offset_left = 0
        x_offset_right = 0
        y_offset_top = 0
        y_offset_bottom = 0

        if (trans_corner > 0).all():
            print('trans positive')
            min_height = min(trans_corner[1])
            max_height = max(trans_corner[1])
            min_width = min(trans_corner[0])
            max_width = max(trans_corner[0])

            if max_width > cols:
                x_offset_right = int(max_width - cols)
            if max_height > row:
                y_offset_bottom = int(max_height - row)

            ref = img_list[i].copy()
            warp_img = img_list[int((i+1)%len(feature_vec_list))].copy()
            ref = cv2.copyMakeBorder(ref, 0, y_offset_bottom, 0, x_offset_right, cv2.BORDER_CONSTANT)
            warp_img = cv2.copyMakeBorder(warp_img, 0, y_offset_bottom, 0, x_offset_right, cv2.BORDER_CONSTANT)
        else:
            print('trans negative')
            H = np.linalg.inv(H)
            trans_corner = H @ corners
            trans_corner = trans_corner/trans_corner[2]

            min_height = min(trans_corner[1])
            max_height = max(trans_corner[1])
            min_width = min(trans_corner[0])
            max_width = max(trans_corner[0])

            if max_width > cols:
                x_offset_right = int(max_width - cols)
            if max_height > row:
                y_offset_bottom = int(max_height - row)

            ref = img_list[int((i+1)%len(feature_vec_list))].copy()
            warp_img = img_list[i].copy()
            ref = cv2.copyMakeBorder(ref, y_offset_top, y_offset_bottom, x_offset_left, x_offset_right, cv2.BORDER_CONSTANT)
            warp_img = cv2.copyMakeBorder(warp_img, y_offset_top, y_offset_bottom, x_offset_left, x_offset_right, cv2.BORDER_CONSTANT)

        # print(trans_corner)
        # print(f'x_offset_left: {x_offset_left}')
        # print(f'x_offset_right: {x_offset_right}')
        # print(f'y_offset_top: {y_offset_top}')
        # print(f'y_offset_bottom: {y_offset_bottom}')

        print(f'final img size: {ref.shape}')
        final_img_rows = ref.shape[0]
        final_img_cols = ref.shape[1]

        warped_img = cv2.warpPerspective(warp_img, H, (final_img_cols, final_img_rows))
        merged_img = ref.copy()
        merged_img[merged_img == 0] = warped_img[merged_img == 0]

        ####################################################################
        #### Blending
        ####################################################################
        mask = np.ones(warped_img.shape, warped_img.dtype) * 255
        mask[warped_img==0] = 0
        mask[ref==0] = 0

        coord = np.where(mask>0)[:2]
        mask_center_pos = np.asarray(np.average(coord, axis=1), dtype=np.int)

        # blend_pos = (int(final_img_cols/2)-24, int(final_img_rows/2)-15)
        blend_pos = tuple((mask_center_pos[1], mask_center_pos[0]))

        print(f'blend pos: {blend_pos}')

        # blend_img = cv2.seamlessClone(warped_img, ref, 
        #                                 mask, 
        #                                 blend_pos, 
        #                                 cv2.NORMAL_CLONE)

        # merged_img[mask > 0] = blend_img[mask>0]

        cv2.imshow('img', warp_img)
        cv2.imshow('img_warped', warped_img)
        cv2.imshow('img_merged', merged_img)
        cv2.imshow('ref', ref)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == ord('q'):
            break



