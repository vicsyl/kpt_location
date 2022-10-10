import torch
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("./superpoint_forked")
from superpoint import SuperPointDescriptor


class SuperPointDetector:

    def __init__(self, path=None, device: torch.device = torch.device('cpu')):
        if not path:
            path = "./superpoint_forked/superpoint_v1.pth"
        self.super_point = SuperPointDescriptor(path, device)

    def detect(self, img_np, mask=None):
        if len(img_np.shape) == 3:
            img_np = cv.cvtColor(img_np, cv.COLOR_RGB2GRAY)
        pts, _, heatmap = self.super_point.detectAndComputeGrey(img_np, mask)
        SuperPointDetector.analyze_heatmap(heatmap, pts)
        return pts

    @staticmethod
    def analyze_heatmap(heatmap, pts):

        def th_heatmap(th):
            return (heatmap >= th).astype(dtype=np.uint8)

        th_heat = th_heatmap(0.0001)

        coords_t = torch.tensor([[round(pt.pt[1]), round(pt.pt[0])] for pt in pts])

        label_count, labels = cv.connectedComponents(th_heat, connectivity=4)

        # plt.figure()
        # plt.imshow(th_heat)
        # plt.show()
        # plt.close()
        # plt.figure()
        # plt.imshow(labels)
        # plt.show()
        # plt.close()
        # heatmap_values = heatmap[coords_t[:, 0], coords_t[:, 1]]
        # hist = np.histogram(heatmap)

        labels_kpts = labels[coords_t[:, 0], coords_t[:, 1]]
        labels_kpts_un, counts = np.unique(labels_kpts, return_counts=True)
        filter_labels = labels_kpts_un[counts > 1]
        valid_labels = labels_kpts_un[counts == 1]

        filter_label_pxs = np.zeros_like(labels)
        cc_only_kpts = np.ones(labels_kpts.shape[0]).astype(dtype=bool)
        for filter_label in filter_labels:
            cc_only_kpts = cc_only_kpts & (labels_kpts != filter_label)
            filter_label_pxs = filter_label_pxs | (labels == filter_label)

        valid_label_pxs = np.zeros_like(labels)
        for valid_label in valid_labels:
            valid_label_pxs = valid_label_pxs | (labels == valid_label)

        print("unique cc: {} out of {}".format(cc_only_kpts.sum(), len(pts)))
        non_valid_kpts = len(pts) - cc_only_kpts.sum()
        non_valid_cc = len(filter_labels)
        print("non_valid_kpts: {}, non_valid_cc: {}, arg.: {}".format(non_valid_kpts, non_valid_cc, non_valid_kpts / non_valid_cc))
        print("invalid label avg. area: {}".format(filter_label_pxs.sum() / non_valid_cc))

        valid_cc = len(valid_labels)
        print("valid label avg. area: {}".format(valid_label_pxs.sum() / valid_cc))

        print("th_heat pxs: {} out of {}".format(th_heat.sum(), th_heat.shape[0] * th_heat.shape[1]))

