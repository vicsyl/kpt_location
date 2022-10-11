import torch
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("./superpoint_forked")
from superpoint import SuperPointDescriptor


def show_plt(img, title):
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    plt.show()
    plt.close()


def show_patch(patch, title, coords):

    patch = patch / patch.max()
    scale = 10
    to_show = cv.resize(patch, dsize=(patch.shape[0] * scale, patch.shape[1] * scale), interpolation=cv.INTER_NEAREST)
    to_show = np.repeat(to_show[:, :, None], 3, axis=2)
    coords = coords * 10
    to_show[coords[:, 0] + scale // 2, coords[:, 1] + scale // 2] = [1.0, 0.0, 0.0]
    show_plt(to_show, title)

    # #show_plt(patch, title)
    # patch = patch / patch.max()
    # patch = np.repeat(patch.copy()[:, :, None], 3, axis=2)
    #patch[coords[:, 0], coords[:, 1]] = [1.0, 0.0, 0.0]
    #show_plt(patch, title)


# def rectify_patch(patch, coords):
#     asc = np.sort(np.ravel(patch[patch > 0.0]))
#     for value in asc:
#         patch = np.where(patch > value, patch, 0.0)
#         if patch[coords[:, 0], coords[:, 1]].min() == 0:
#             title = "rect. patch coord missed for th={}, kpts locations: {}".format(value, len(coords))
#             show_patch(patch, title, coords)
#             break
#         else:
#             thd = (patch > 0.0).astype(dtype=np.uint8)
#             label_count, labels = cv.connectedComponents(thd, connectivity=4)
#             if label_count > 2:
#                 title = "patch fully rectified for th={}, kpts locations: {}".format(value, len(coords))
#                 show_patch(patch, title, coords)
#                 return
#             else:
#                 pass
#     #             title = "patch rectified for th={}, kpts locations: {}".format(value, len(coords))
#     #             show_patch(patch, title, coords)
#     print("PATCH rectrification finished")


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
        self.get_hm_patches(heatmap, pts_cv)
        return pts_cv

    def th_heatmap_mask(self, heatmap, print_out=True):
        ret = (heatmap >= self.heat_map_th).astype(dtype=np.uint8)
        if print_out:
            print("thresholded heat map area: {:.04f}".format(ret.sum() / (heatmap.shape[0] * heatmap.shape[1])))
        return ret

    # def get_hm_patches_old(self, heatmap, pts_cv):
    #
    #     # NOTE a little bit redundant
    #     coords_t = torch.tensor([[round(pt.pt[1]), round(pt.pt[0])] for pt in pts_cv])
    #
    #     th_heat_mask = self.th_heatmap_mask(heatmap)
    #     label_count, labels = cv.connectedComponents(th_heat_mask, connectivity=4)
    #
    #     labels_kpts = labels[coords_t[:, 0], coords_t[:, 1]]
    #     all_kpt_labels, counts = np.unique(labels_kpts, return_counts=True)
    #     filter_labels = all_kpt_labels[counts > 1]
    #     valid_labels = all_kpt_labels[counts == 1]
    #
    #     for label in all_kpt_labels:
    #         grid = np.where(labels == label)
    #         y_min, y_max = grid[0].min(), grid[0].max()
    #         x_min, x_max = grid[1].min(), grid[1].max()
    #         patch = np.where(labels == label, heatmap, 0.0)
    #         indices = np.where(labels_kpts == label)
    #         coords = coords_t[indices]
    #         if patch[coords[:, 0], coords[:, 1]].min() == 0:
    #             pass
    #         #show_plt(patch, "whole patch: {}".format(coords))
    #         coords = coords - torch.tensor([y_min, x_min])
    #         patch = patch[y_min:y_max + 1, x_min:x_max + 1]
    #         if patch[coords[:, 0], coords[:, 1]].min() == 0:
    #             pass
    #         # if coords.shape[0] > 1:
    #         #     show_patch(patch, "heatmap not isolated patch kpts locations: {}".format(len(coords)), coords)
    #         #     #rectify_patch(patch, coords)
    #         # else:
    #         #     pass
    #         #     #show_patch(patch, "heatmap isolated patch kpts locations: {} pxs".format(grid[0].shape[0]), coords)

    def get_hm_patches(self, heatmap, pts_cv):

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

        non_valid_kpts = len(pts_cv) - len(valid_labels)
        non_valid_cc = len(invalid_labels)
        print("invalid kpts: {}, invalid components: {}, avg kpts. per component: {}".format(non_valid_kpts, non_valid_cc, non_valid_kpts / non_valid_cc))
        print("invalid component avg. area: {}".format(filter_label_pxs.sum() / non_valid_cc))

        max_y_margin = 0
        max_x_margin = 0
        y_margins = np.zeros_like(labels_kpts)
        x_margins = np.zeros_like(labels_kpts)

        for i, label in enumerate(labels_kpts):

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

            if i < 8:
                coord = coord - torch.tensor([y_min, x_min])
                patch = heatmap[y_min:y_max + 1, x_min:x_max + 1]
                show_patch(patch, "example", coord[None])

        assert np.all(y_margins >= 0)
        assert np.all(x_margins >= 0)
        print(f"max x margin: {max_x_margin}")
        print(f"max y margin: {max_y_margin}")