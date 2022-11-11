import cv2 as cv
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import math

from PIL import Image

from utils import get_tentatives


def draw_all_kpts(img_np, kpts_original, kpts_transformed, title):
    for kpt in kpts_original:
        kpt.size = 50
        kpt.angle = math.pi / 2
    for kpt in kpts_transformed:
        kpt.size = 25
        kpt.angle = 0
    kpts_all_cv = kpts_original + kpts_transformed
    img_kpts = img_np.copy()
    cv.drawKeypoints(img_kpts, kpts_all_cv, img_kpts, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(9, 9))
    title = "{}: {} - transformed of size 25, original of size 50".format(title, len(kpts_all_cv))
    plt.title(title)
    plt.imshow(img_kpts)
    plt.show()
    plt.close()


def mnn_generic(pts1, pts2, err_th):
    assert len(pts1.shape) == len(pts2.shape)

    pts1_r, pts2_r = pts1, pts2
    if len(pts1.shape) == 1:
        pts1_r, pts2_r = pts1[:, None].repeat(1, 2), pts2[:, None].repeat(1, 2)
        pts1_r[:, 1] = 0.0
        pts2_r[:, 1] = 0.0

    d_mat = torch.cdist(pts1_r, pts2_r)

    min0_values, min0_indices = torch.min(d_mat, dim=0)
    min1_values, min1_indices = torch.min(d_mat, dim=1)

    mask = min1_indices[min0_indices] == torch.arange(0, min0_indices.shape[0])
    mask2_boolean = mask & (min0_values < err_th)

    verify = True
    if verify:
        for i in range(min0_indices.shape[0]):
            if mask2_boolean[i]:
                assert min1_indices[min0_indices[i]] == i
                assert min0_values[i] < err_th

    mask1 = min0_indices[mask2_boolean]
    pts1_new = pts1[mask1]
    pts2_new = pts2[mask2_boolean]

    mask2 = mask2_boolean.nonzero()[:, 0]
    return pts1_new, pts2_new, mask1, mask2


def detect_robust(detector, img_np):
    kpts = detector.detect(img_np, mask=None)
    if len(kpts) == 2:
        kpts = kpts[0]
    return kpts


def rotate_experiment(file_path, detector, rotations_90_deg, err_th, show_img=True, use_mnn=True):
    print(f"experiment for n rotations: {rotations_90_deg}")

    img_np_o = np.array(Image.open(file_path))
    new_h, new_w = img_np_o.shape[0] // 8 * 8, img_np_o.shape[1] // 8 * 8
    img_np_o = img_np_o[:new_h, :new_w]

    kpts_0_cv, desc_0_cv = detector.detectAndCompute(img_np_o, None)
    kpts_0 = torch.tensor([[kp.pt[1], kp.pt[0]] for kp in kpts_0_cv])

    sizes_0 = np.array([k.size for k in kpts_0_cv])
    print(np.histogram(sizes_0))

    img_np_r = np.rot90(img_np_o, rotations_90_deg, [0, 1])

    kpts_1_cv, desc_1_cv = detector.detectAndCompute(img_np_r, None)
    kpts_1 = torch.tensor([[kp.pt[1], kp.pt[0]] for kp in kpts_1_cv])
    sizes_1 = np.array([k.size for k in kpts_1_cv])

    coord0_max = img_np_r.shape[0] - 1
    coord1_max = img_np_r.shape[1] - 1
    for i in range(4 - rotations_90_deg):
        kpts_10_new = coord1_max - kpts_1[:, 1]
        kpts_11_new = kpts_1[:, 0].clone()
        kpts_1[:, 0] = kpts_10_new.clone()
        kpts_1[:, 1] = kpts_11_new.clone()
        coord1_max, coord0_max = coord0_max, coord1_max

    for i, e_kpts_1_cv in enumerate(kpts_1_cv):
        t = e_kpts_1_cv.pt
        e_kpts_1_cv.pt = (kpts_1[i, 1].item(), kpts_1[i, 0].item())

    print("number of kpts: {}, {}".format(kpts_0.shape[0], kpts_1.shape[0]))

    if use_mnn:
        # mnn
        kpts_0_new_1d, kpts_1_new_1d, mask_00, mask_10 = mnn_generic(kpts_0, kpts_1, err_th=err_th)
    else:
        kpts_0_new_1d_mmn, kpts_1_new_1d_mmn, mask_f00, mask_f10 = mnn_generic(kpts_0, kpts_1, err_th=err_th)
        ratio_threshold = 0.8
        kpts_0_new_1d, kpts_1_new_1d, _, _, tentative_matches = get_tentatives(kpts_0_cv, desc_0_cv, kpts_1_cv, desc_1_cv, ratio_threshold=ratio_threshold, space_dist_th=3.0)
        kpts_0_new_1d = torch.from_numpy(kpts_0_new_1d)
        kpts_1_new_1d = torch.from_numpy(kpts_1_new_1d)
        mask_00 = torch.tensor([t.queryIdx for t in tentative_matches])
        mask_10 = torch.tensor([t.trainIdx for t in tentative_matches])

    print("number of filtered kpts: {}, {}".format(kpts_0_new_1d.shape[0], kpts_1_new_1d.shape[0]))

    distances = torch.linalg.norm(kpts_1_new_1d - kpts_0_new_1d, axis=1)
    print("distances between matching keypoints - min: {}, max: {}".format(distances.min(), distances.max()))

    def get_lists(i_l, torch_mask):
        m_l = torch_mask.tolist()
        matched_l, unmatched_l = [], []
        for index, item in enumerate(i_l):
            if index in m_l:
                matched_l.append(item)
            else:
                unmatched_l.append(item)
        return matched_l, unmatched_l

    # visualize the matched and unmatched keypoints
    kpts_0_cv_matched, kpts_0_cv_unmatched = get_lists(kpts_0_cv, mask_00)
    kpts_1_cv_matched, kpts_1_cv_unmatched = get_lists(kpts_1_cv, mask_10)
    print("Matched keypoints: {}".format(len(kpts_0_cv_matched) + len(kpts_1_cv_matched)))
    print("Unmatched keypoints: {}".format(len(kpts_0_cv_unmatched) + len(kpts_1_cv_unmatched)))
    # if show_img:
    #     draw_all_kpts(img_np_o, kpts_0_cv_matched, kpts_1_cv_matched, "matched keypoints")
    #     draw_all_kpts(img_np_o, kpts_0_cv_unmatched, kpts_1_cv_unmatched, "unmatched keypoints")

    # TODO rename
    mean = (kpts_1_new_1d - kpts_0_new_1d).mean(dim=0)
    var = (kpts_1_new_1d - kpts_0_new_1d).var(dim=0)
    stad_dev = var.sqrt()
    print("mean: {}, variance: {}, std dev: {}".format(mean.numpy(), var.numpy(), stad_dev.numpy()))
    print()

    def print_stats(pt1, pt2):
        pt1 = torch.from_numpy(pt1)
        pt2 = torch.from_numpy(pt2)
        mean = (pt2 - pt1).mean(dim=0)
        var = (pt2 - pt1).var(dim=0)
        stad_dev = var.sqrt()
        print("mean: {}, variance: {}, std dev: {}".format(mean.numpy(), var.numpy(), stad_dev.numpy()))

    kpts_0_all = np.array([kpts_0_cv[m].pt for m in mask_00])
    kpts_1_all = np.array([kpts_1_cv[m].pt for m in mask_10])

    print("All stats")
    print_stats(kpts_0_all, kpts_1_all)

    sizes_00 = np.array([kpts_0_cv[m].size for m in mask_00])
    hist = np.histogram(sizes_0)

    for i in range(1, len(hist[1])):
        lower = hist[1][i - 1]
        upper = hist[1][i]
        if i != len(hist[1]) - 1:
            mask = np.logical_and(lower <= sizes_00, sizes_00 < upper)
        else:
            mask = np.logical_and(lower <= sizes_00, sizes_00 <= upper)
        count = hist[0][i - 1]
        if count == 0:
            print(f"\nno kpts between sizes [{lower}, {upper}] ({count} kpts.)")
            continue
        print(f"\nstats for kpts between sizes [{lower}, {upper}] ({count} kpts.)")
        kpts_0_filtered = np.array([kpts_0_all[i] for i, m in enumerate(mask) if m])
        kpts_1_filtered = np.array([kpts_1_all[i] for i, m in enumerate(mask) if m])
        print_stats(kpts_0_filtered, kpts_1_filtered)


def rotate_experiment_loop(detector, img_to_show, err_th, show_img=True, use_mnn=False):
    img_dir = "demo_imgs/bark"
    files = ["{}/{}".format(img_dir, fn) for fn in os.listdir(img_dir)][:img_to_show]
    for file_path in files:
        print(f"\n\nFILE: {file_path}\n")
        for rots in range(1, 4):
            rotate_experiment(file_path, detector, rots, err_th, show_img, use_mnn)


if __name__ == "__main__":
    from superpoint_local import SuperPointDetector
    from sift_detectors import AdjustedSiftDescriptor
    detector = AdjustedSiftDescriptor(adjustment=[0., 0.])
    rotate_experiment_loop(detector, img_to_show=6, err_th=4)
