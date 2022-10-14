import torch
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("./superpoint_forked")
from superpoint import SuperPointDescriptor

from utils import show_np


def show_patch(patch, title, coords):

    patch = patch / patch.max()
    scale = 10
    to_show = cv.resize(patch, dsize=(patch.shape[1] * scale, patch.shape[0] * scale), interpolation=cv.INTER_NEAREST)
    to_show = np.repeat(to_show[:, :, None], 3, axis=2)
    coords = coords * 10
    to_show[coords[:, 0] + scale // 2, coords[:, 1] + scale // 2] = [1.0, 0.0, 0.0]
    show_np(to_show, title)


class SuperPointDetector:

    def __init__(self, path=None, device: torch.device = torch.device('cpu')):
        if not path:
            path = "./superpoint_forked/superpoint_v1.pth"
        self.super_point = SuperPointDescriptor(path, device)
        self.heat_map_th = self.super_point.sp_frontend.conf_thresh

    def detect(self, img_np, mask=None):
        if len(img_np.shape) == 3:
            img_np = cv.cvtColor(img_np, cv.COLOR_RGB2GRAY)
        pts, _, heatmap = self.super_point.detectAndComputeGrey(img_np, mask)
        pts_cv = [cv.KeyPoint(pt[0], pt[1], 1) for pt in pts]
        analyze = False
        if analyze:
            self.analyze_hm_patches(heatmap, pts_cv)
        return pts_cv, heatmap

    def th_heatmap_mask(self, heatmap, print_out=True):
        ret = (heatmap >= self.heat_map_th).astype(dtype=np.uint8)
        if print_out:
            print("thresholded heat map area: {:.04f}".format(ret.sum() / (heatmap.shape[0] * heatmap.shape[1])))
        return ret

    def analyze_hm_patches(self, heatmap, pts_cv):

        coords_t = torch.tensor([[round(pt.pt[1]), round(pt.pt[0])] for pt in pts_cv])

        th_heat_mask = self.th_heatmap_mask(heatmap)
        label_count, labels = cv.connectedComponents(th_heat_mask, connectivity=4)

        labels_kpts = labels[coords_t[:, 0], coords_t[:, 1]]
        all_kpt_labels, counts = np.unique(labels_kpts, return_counts=True)
        invalid_labels = all_kpt_labels[counts > 1]
        valid_labels = all_kpt_labels[counts == 1]

        filter_label_pxs = np.zeros_like(labels)
        for filter_label in invalid_labels:
            filter_label_pxs = filter_label_pxs | (labels == filter_label)

        valid_label_pxs = np.zeros_like(labels)
        for valid_label in valid_labels:
            valid_label_pxs = valid_label_pxs | (labels == valid_label)

        print("isolated kpts: {} out of {}".format(len(valid_labels), len(pts_cv)))
        print("avg. area of component for isolated kpts: {}".format(valid_label_pxs.sum() / len(valid_labels)))

        invalid_kpts = len(pts_cv) - len(valid_labels)
        invalid_cc = len(invalid_labels)
        if invalid_kpts == 0:
            print("no invalid kpts!")
        else:
            print("invalid kpts: {}, invalid components: {}, avg kpts. per component: {}".format(invalid_kpts, invalid_cc, invalid_kpts / invalid_cc))
            print("invalid component avg. area: {}".format(filter_label_pxs.sum() / invalid_cc))

        max_y_margin = 0
        max_x_margin = 0
        y_margins = np.zeros_like(labels_kpts)
        x_margins = np.zeros_like(labels_kpts)

        show_patches = 1
        for i, label in enumerate(labels_kpts[:show_patches]):

            cc_grid = np.where(labels == label)
            y_min, y_max = cc_grid[0].min(), cc_grid[0].max()
            x_min, x_max = cc_grid[1].min(), cc_grid[1].max()
            # bb_box = (y_max - y_min + 1, x_max - x_min + 1)
            # ... bb_box stat?

            coord = coords_t[i]
            y_margin = max(y_max - coord[0], coord[0] - y_min)
            y_margins[i] = y_margin
            if y_margin > max_y_margin:
                max_y_margin = y_margin.item()

            x_margin = max(x_max - coord[1], coord[1] - x_min)
            x_margins[i] = x_margin
            if x_margin > max_x_margin:
                max_x_margin = x_margin.item()

            coord = coord - torch.tensor([y_min, x_min])
            patch = heatmap[y_min:y_max + 1, x_min:x_max + 1]
            show_patch(patch, "example", coord[None])

        assert np.all(y_margins >= 0)
        assert np.all(x_margins >= 0)
        print(f"max x margin: {max_x_margin}")
        print(f"max y margin: {max_y_margin}")