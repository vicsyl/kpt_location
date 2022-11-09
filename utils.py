import math

import matplotlib.pyplot as plt
import torch
import cv2 as cv
import numpy as np


def show_np(img, title):
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    plt.show()
    plt.close()


def show_torch(img, title):
    if len(img.shape) == 4:
        img = img[0]
    if img.shape[0] == 1:
        img = img[0]
    show_np(img.numpy(), title)


def avg_loss(data):
    return (torch.linalg.norm(data, dim=1) ** 2).sum() / data.shape[0]


def split_points(tentative_matches, kps0, kps1):
    src_pts = np.float32([kps0[m.queryIdx].pt for m in tentative_matches]).reshape(-1, 2)
    dst_pts = np.float32([kps1[m.trainIdx].pt for m in tentative_matches]).reshape(-1, 2)
    kps0 = [kps0[m.queryIdx] for m in tentative_matches]
    kps1 = [kps1[m.trainIdx] for m in tentative_matches]
    return src_pts, dst_pts, kps0, kps1


def get_tentatives(kpts0, desc0, kpts1, desc1, ratio_threshold, space_dist_th=None):
    matcher = cv.BFMatcher(crossCheck=False)
    knn_matches = matcher.knnMatch(desc0, desc1, k=2)
    matches2 = matcher.match(desc1, desc0)

    tentative_matches = []
    for m, n in knn_matches:
        if matches2[m.trainIdx].trainIdx != m.queryIdx:
            continue

        if space_dist_th:
            x = kpts0[m.queryIdx].pt
            y = kpts1[m.trainIdx].pt
            dist = math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
            if dist > space_dist_th:
                continue

        if m.distance < ratio_threshold * n.distance:
            tentative_matches.append(m)

    src, dst, kpts0, kpts1 = split_points(tentative_matches, kpts0, kpts1)
    return src, dst, kpts0, kpts1, tentative_matches
