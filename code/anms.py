import cv2
import numpy as np

def anms(corners, N_req, corner_fn):
    infinite = 1000000
    r = np.zeros_like(corner_fn)
    for i in corners:
        x,y = i.ravel()
        r[x,y] = infinite
    c_robust = 0.9

    for i in range(corners.shape[0]):
        for j in range(corners.shape[0]):
            dist = infinite

            if corner_fn[tuple(corners[i,:])] < c_robust*corner_fn[tuple(corners[j,:])]:
                dist = np.sum(np.square(corners[i,:] - corners[j,:]))
            else:
                continue

            if dist < r[tuple(corners[i,:])]:
                r[tuple(corners[i,:])] = dist

    sorted_elements_idx = np.dstack(np.unravel_index(np.argsort(r, axis=None), r.shape))
    sorted_elements_idx = np.squeeze(sorted_elements_idx)

    n_req_corners = sorted_elements_idx[-N_req:, :]
    return n_req_corners

def anms_vectorised(corners, N_req, corner_fn):
    infinite = 1000000
    c_robust = 0.9
    r = np.zeros_like(corner_fn)

    for i in corners:
        x,y = i.ravel()
        r[x,y] = infinite

    corners_tuple = tuple((corners[:,0], corners[:,1]))
    corner_mask = np.zeros_like(corner_fn)
    corner_mask[corners_tuple] = corner_fn[corners_tuple]

    for i in range(corners.shape[0]):
        # min_neigh_pt = (1.0/c_robust) * corner_fn[tuple(corners[i,:])]
        # pts = np.squeeze(np.dstack(np.where(corner_mask > min_neigh_pt)))
        pts = np.vstack(np.where(c_robust*corner_mask > corner_fn[tuple(corners[i,:])]))

        if pts.shape[1] == 0:
            continue

        dist = np.sum(np.square(pts.T - corners[i]), axis=1)
        r[tuple(corners[i,:])] = dist.min()

    sorted_elements_idx = np.dstack(np.unravel_index(np.argsort(r, axis=None), r.shape))
    sorted_elements_idx = np.squeeze(sorted_elements_idx)

    n_req_corners = sorted_elements_idx[-N_req:, :]
    return n_req_corners.astype(np.float32), np.argsort(r, axis=None)[-N_req:]
