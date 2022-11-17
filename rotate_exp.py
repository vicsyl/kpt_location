import cv2 as cv
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import math

import PIL
from PIL import Image

from utils import get_tentatives
from prepare_data import scale_pil


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


def scale_experiment_against(kpts_0_cv, desc_0_cv, file_path, detector, scale, show_img=True):

    img_pil = Image.open(file_path)

    img_pil, scale = scale_pil(img_pil, scale, {"integer_scale": True}, show=False)
    img_np_r = np.array(img_pil)

    kpts_1_cv, desc_1_cv = detector.detectAndCompute(img_np_r, None)
    kpts_1_geo = torch.tensor([[kp.pt[1], kp.pt[0]] for kp in kpts_1_cv])
    kpts_1_geo = kpts_1_geo / scale

    for i, e_kpts_1_cv in enumerate(kpts_1_cv):
        e_kpts_1_cv.pt = (kpts_1_geo[i, 1].item(), kpts_1_geo[i, 0].item())

    ratio_threshold = 0.8
    kpts_0_new_geo, kpts_1_new_geo, _, _, tentative_matches = get_tentatives(kpts_0_cv, desc_0_cv, kpts_1_cv, desc_1_cv, ratio_threshold=ratio_threshold, space_dist_th=5.0)
    mask_00 = torch.tensor([t.queryIdx for t in tentative_matches])
    mask_10 = torch.tensor([t.trainIdx for t in tentative_matches])

    return mask_00, mask_10, kpts_1_geo


def rotate_experiment_against(kpts_0_cv, desc_0_cv, file_path, detector, rotations_90_deg, show_img=True):

    # print(f"experiment for n rotations: {rotations_90_deg}")

    img_np_o = np.array(Image.open(file_path))
    new_h, new_w = img_np_o.shape[0] // 8 * 8, img_np_o.shape[1] // 8 * 8
    img_np_o = img_np_o[:new_h, :new_w]

    kpts_0 = torch.tensor([[kp.pt[1], kp.pt[0]] for kp in kpts_0_cv])
    sizes_0 = np.array([k.size for k in kpts_0_cv])
    # print(np.histogram(sizes_0))

    img_np_r = np.rot90(img_np_o, rotations_90_deg, [0, 1])

    kpts_1_cv, desc_1_cv = detector.detectAndCompute(img_np_r, None)
    kpts_1_geo = torch.tensor([[kp.pt[1], kp.pt[0]] for kp in kpts_1_cv])

    coord0_max = img_np_r.shape[0] - 1
    coord1_max = img_np_r.shape[1] - 1
    for i in range(4 - rotations_90_deg):
        kpts_10_new = coord1_max - kpts_1_geo[:, 1]
        kpts_11_new = kpts_1_geo[:, 0].clone()
        kpts_1_geo[:, 0] = kpts_10_new.clone()
        kpts_1_geo[:, 1] = kpts_11_new.clone()
        coord1_max, coord0_max = coord0_max, coord1_max

    for i, e_kpts_1_cv in enumerate(kpts_1_cv):
        e_kpts_1_cv.pt = (kpts_1_geo[i, 1].item(), kpts_1_geo[i, 0].item())

    # print("number of kpts: {}, {}".format(kpts_0.shape[0], kpts_1_geo.shape[0]))

    ratio_threshold = 0.8
    kpts_0_new_geo, kpts_1_new_geo, _, _, tentative_matches = get_tentatives(kpts_0_cv, desc_0_cv, kpts_1_cv, desc_1_cv, ratio_threshold=ratio_threshold, space_dist_th=3.0)
    # kpts_0_new_geo = torch.from_numpy(kpts_0_new_geo)
    # kpts_1_new_geo = torch.from_numpy(kpts_1_new_geo)
    mask_00 = torch.tensor([t.queryIdx for t in tentative_matches])
    mask_10 = torch.tensor([t.trainIdx for t in tentative_matches])

    return mask_00, mask_10, kpts_1_geo


def rotate_experiment(file_path, detector, rotations_90_deg, err_th, show_img=True, use_mnn=True):
    print(f"experiment for n rotations: {rotations_90_deg}")

    img_np_o = np.array(Image.open(file_path))
    new_h, new_w = img_np_o.shape[0] // 8 * 8, img_np_o.shape[1] // 8 * 8
    img_np_o = img_np_o[:new_h, :new_w]

    kpts_0_cv, desc_0_cv = detector.detectAndCompute(img_np_o, None)

    img_np_r = np.rot90(img_np_o, rotations_90_deg, [0, 1])

    kpts_1_cv, desc_1_cv = detector.detectAndCompute(img_np_r, None)

    rotate_exp_from_kpts(img_np_o, img_np_r, rotations_90_deg, kpts_0_cv, desc_0_cv, kpts_1_cv, desc_1_cv, use_mnn, err_th, show_img)


def rotate_exp_from_kpts(img_np_o, img_np_r, rotations_90_deg, kpts_0_cv, desc_0_cv, kpts_1_cv, desc_1_cv, use_mnn, err_th, show_img):

    kpts_0 = torch.tensor([[kp.pt[1], kp.pt[0]] for kp in kpts_0_cv])
    sizes_0 = np.array([k.size for k in kpts_0_cv])
    # print(np.histogram(sizes_0))

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
        kpts_0_new_geo, kpts_1_new_geo, mask_00, mask_10 = mnn_generic(kpts_0, kpts_1, err_th=err_th)
    else:
        ratio_threshold = 0.8
        kpts_0_new_geo, kpts_1_new_geo, _, _, tentative_matches = get_tentatives(kpts_0_cv, desc_0_cv, kpts_1_cv, desc_1_cv, ratio_threshold=ratio_threshold, space_dist_th=3.0)
        kpts_0_new_geo = torch.from_numpy(kpts_0_new_geo)
        kpts_1_new_geo = torch.from_numpy(kpts_1_new_geo)
        mask_00 = torch.tensor([t.queryIdx for t in tentative_matches])
        mask_10 = torch.tensor([t.trainIdx for t in tentative_matches])

    print("number of filtered kpts: {}, {}".format(kpts_0_new_geo.shape[0], kpts_1_new_geo.shape[0]))

    distances = torch.linalg.norm(kpts_1_new_geo - kpts_0_new_geo, axis=1)
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

    # NOTE that this changes the kpts' sizes
    if show_img:
        if len(img_np_o.shape) == 2:
            img_np_o_c = cv.cvtColor(img_np_o, cv.COLOR_GRAY2RGB)
        else:
            img_np_o_c = img_np_o
        draw_all_kpts(img_np_o_c, kpts_0_cv_matched, kpts_1_cv_matched, "matched keypoints")
        draw_all_kpts(img_np_o_c, kpts_0_cv_unmatched, kpts_1_cv_unmatched, "unmatched keypoints")

    # TODO rename
    mean = (kpts_1_new_geo - kpts_0_new_geo).mean(dim=0)
    var = (kpts_1_new_geo - kpts_0_new_geo).var(dim=0)
    stad_dev = var.sqrt()
    print("mean: {}, variance: {}, std dev: {}".format(mean.numpy(), var.numpy(), stad_dev.numpy()))
    print()

    # def print_stats(pt1, pt2):
    #     pt1 = torch.from_numpy(pt1)
    #     pt2 = torch.from_numpy(pt2)
    #     mean = (pt2 - pt1).mean(dim=0)
    #     var = (pt2 - pt1).var(dim=0)
    #     stad_dev = var.sqrt()
    #     print("mean: {}, variance: {}, std dev: {}".format(mean.numpy(), var.numpy(), stad_dev.numpy()))
    #
    # kpts_0_all = np.array([kpts_0_cv[m].pt for m in mask_00])
    # kpts_1_all = np.array([kpts_1_cv[m].pt for m in mask_10])
    #
    # print("All stats")
    # print_stats(kpts_0_all, kpts_1_all)
    #
    # sizes_00 = np.array([kpts_0_cv[m].size for m in mask_00])
    # hist = np.histogram(sizes_0)

    # for i in range(1, len(hist[1])):
    #     lower = hist[1][i - 1]
    #     upper = hist[1][i]
    #     if i != len(hist[1]) - 1:
    #         mask = np.logical_and(lower <= sizes_00, sizes_00 < upper)
    #     else:
    #         mask = np.logical_and(lower <= sizes_00, sizes_00 <= upper)
    #     count = hist[0][i - 1]
    #     if count == 0:
    #         print(f"\nno kpts between sizes [{lower}, {upper}] ({count} kpts.)")
    #         continue
    #     print(f"\nstats for kpts between sizes [{lower}, {upper}] ({count} kpts.)")
    #     kpts_0_filtered = np.array([kpts_0_all[i] for i, m in enumerate(mask) if m])
    #     kpts_1_filtered = np.array([kpts_1_all[i] for i, m in enumerate(mask) if m])
    #     print_stats(kpts_0_filtered, kpts_1_filtered)


def rotate_detail_experiment_loop(detector, img_to_show, show_img=True):

    def open_img(f_p):
        img_np_o = np.array(Image.open(f_p))
        new_h, new_w = img_np_o.shape[0] // 8 * 8, img_np_o.shape[1] // 8 * 8
        img_np_o = img_np_o[:new_h, :new_w]
        return img_np_o

    img_dir = "demo_imgs/hypersim"
    files = ["{}/{}".format(img_dir, fn) for fn in os.listdir(img_dir)][:img_to_show]
    for file_path in files:
        print(f"\n\nFILE: {file_path}\n")

        img_np_o = open_img(file_path)

        kpts_0_cv, desc_0_cv = detector.detectAndCompute(img_np_o, None)
        kpts_0_geo = torch.tensor([[kp.pt[1], kp.pt[0]] for kp in kpts_0_cv])
        #kpts_geos = [kpts_0_geo]

        kpts_geos = torch.zeros(len(kpts_0_cv), 4, 2)
        kpts_geos[:, 0, :] = kpts_0_geo

        mask_00 = torch.ones(len(kpts_0_cv), dtype=bool)
        for rots in range(1, 4):
            mask_00_and, mask_10, kpts_1_geo = rotate_experiment_against(kpts_0_cv, desc_0_cv, file_path, detector, rots, show_img)
            mask_00_bool_and = torch.zeros(len(kpts_0_cv), dtype=bool)
            mask_00_bool_and[mask_00_and] = True
            mask_00 = mask_00 & mask_00_bool_and
            kpts_geos[mask_00_and, rots, :] = kpts_1_geo[mask_10]

        kpts_geos = kpts_geos[mask_00]

        margin = 11
        for kpt in kpts_geos[:3]:
            kpt_mean = kpt.mean(dim=0)
            kpt_int = torch.round(kpt_mean)
            if kpt_int[0].item() < margin or kpt_int[0].item() > img_np_o.shape[0] - 1 - margin:
                continue
            if kpt_int[1].item() < margin or kpt_int[1].item() > img_np_o.shape[1] - 1 - margin:
                continue
            show_me_rotate(kpt, img_np_o)


def prepare_lowe(img_to_show):

    def open_img(f_p):
        img_np_o = np.array(Image.open(f_p))
        new_h, new_w = img_np_o.shape[0] // 8 * 8, img_np_o.shape[1] // 8 * 8
        img_np_o = img_np_o[:new_h, :new_w]
        return img_np_o

    img_dir = "demo_imgs/hypersim"
    files = ["{}/{}".format(img_dir, fn) for fn in os.listdir(img_dir)][:img_to_show]
    for i, file_path in enumerate(files):
        img_np = open_img(file_path)
        img_np = cv.cvtColor(img_np, cv.COLOR_RGB2GRAY)
        cv.imwrite(f"demo_imgs/rotated/{i}_rot_0.pgm", img_np)
        for j in range(4):
            img_np = np.rot90(img_np, 1, [0, 1])
            cv.imwrite(f"demo_imgs/rotated/{i}_rot_{j}.pgm", img_np)


def scale_detail_experiment_loop(detector, img_to_show, show_img=True):

    all_kpts = [np.zeros((0, 2)) for _ in range(10)]

    def open_img(f_p):
        img_np_o = np.array(Image.open(f_p))
        new_h, new_w = img_np_o.shape[0] // 8 * 8, img_np_o.shape[1] // 8 * 8
        img_np_o = img_np_o[:new_h, :new_w]
        return img_np_o

    img_dir = "demo_imgs/hypersim"
    files = ["{}/{}".format(img_dir, fn) for fn in os.listdir(img_dir)][:img_to_show]
    for file_path in files:
        print(f"\n\nFILE: {file_path}\n")

        img_np_o = open_img(file_path)

        kpts_0_cv, desc_0_cv = detector.detectAndCompute(img_np_o, None)
        kpts_0_geo = torch.tensor([[kp.pt[1], kp.pt[0]] for kp in kpts_0_cv])
        #kpts_geos = [kpts_0_geo]

        kpts_geos = -torch.ones(len(kpts_0_cv), 10, 2)
        kpts_geos_bool = torch.zeros(len(kpts_0_cv), 10)
        kpts_geos[:, 0, :] = kpts_0_geo
        kpts_geos_bool[:, 0] = True

        mask_00 = torch.ones(len(kpts_0_cv), dtype=bool)
        for scale_int in range(1, 10):
            scale = scale_int / 10
            mask_00_and, mask_10, kpts_1_geo = scale_experiment_against(kpts_0_cv, desc_0_cv, file_path, detector, scale, show_img)
            if mask_00_and.sum().item() > 0:
                mask_00_bool_and = torch.zeros(len(kpts_0_cv), dtype=bool)
                mask_00_bool_and[mask_00_and] = True
                mask_00 = mask_00 & mask_00_bool_and
                # 10 - scale_int => so that original is at 0 and then scale 0.9, 0.8, etc.
                kpts_geos[mask_00_and, 10 - scale_int, :] = kpts_1_geo[mask_10]
                kpts_geos_bool[mask_00_and, 10 - scale_int] = True

        kpts_geos_bool_sums = kpts_geos_bool.sum(dim=1)

        th = 9
        kpts_geos = kpts_geos[kpts_geos_bool_sums > th]
        kpts_geos_bool = kpts_geos_bool[kpts_geos_bool_sums > th]
        # if mask_00.sum() == 0:

        margin = 21
        for i, kpt in enumerate(kpts_geos[:1]):
            kpt_mean = kpt[0]
            kpt_int = torch.round(kpt_mean)
            show = True
            if kpt_int[0].item() < margin or kpt_int[0].item() > img_np_o.shape[0] - 1 - margin:
                show = False
            if kpt_int[1].item() < margin or kpt_int[1].item() > img_np_o.shape[1] - 1 - margin:
                show = False
            mask = kpts_geos_bool[i]
            if show:
                show_me_scale(kpt, mask, img_np_o, margin=margin)
            kpt = kpt - kpt_mean
            scales = torch.from_numpy(np.arange(start=1.0, stop=0.0, step=-0.1))
            kpt = scales[:, None] * kpt
            for i in range(10):
                if mask[i]:
                    all_kpts[i] = np.vstack((all_kpts[i], kpt[None, i]))
            #all_kpts = np.vstack((all_kpts, kpt[None]))

    for i in range(10):
        m = all_kpts[i].mean(axis=0)
        v = all_kpts[i].var(axis=0)
        print(f"i: {i}/{all_kpts[i].shape[0]}: mean: {m}, variance: {v}")


counter_img = 0


def show_me_rotate(kpt, img_np, margin=11):

    counter = 0
    global counter_img
    counter_img += 1

    print(f"counter: {counter_img}")
    scale_crop = 5
    scale_err = 10

    def show(img, title):
        img = img.copy()
        plt.figure(figsize=(4, 4))
        bound = margin / scale_err

        quarters = round(bound / 0.25)
        labels = [0.25 * i for i in range(-quarters, quarters + 1)]
        tickz = [12.5 * i + 55 for i in range(-quarters, quarters + 1)]

        plt.xticks(tickz, labels) #, rotation='vertical')
        plt.yticks(tickz, labels)
        plt.xlabel("px")
        plt.ylabel("px", rotation="horizontal")

        # plt.xlim(-bound, bound)
        plt.imshow(img)
        # plt.title(title)
        # plt.subplots_adjust(left=0.15, bottom=-0.2, right=1.0, top=1.4, wspace=200, hspace=100)
        plt.subplots_adjust(left=0.15, bottom=0.10, right=1.0, top=1.0, wspace=0, hspace=0)
        plt.savefig(f"./visualizations/{counter_img}_{counter}.png")
        plt.show()
        plt.close()

    kpt_mean = kpt.mean(dim=0)
    kpt = kpt - kpt_mean
    kpt_int = torch.round(kpt_mean).to(dtype=int).tolist()
    # kpt_mean_r = kpt_mean - kpt_int

    fig = plt.figure()

    left = (kpt_int[1] - 50)
    right = (kpt_int[1] + 50)
    if left < 0:
        right = right - left
        left = 0
    top = (kpt_int[0] - 50)
    bottom = (kpt_int[0] + 50)
    if top < 0:
        bottom = bottom - top
        top = 0

    start = (kpt_int[1] - margin - left, kpt_int[0] - margin - top)
    end = (kpt_int[1] + margin - left, kpt_int[0] + margin - top)

    # start = (kpt_int[1] - margin, kpt_int[0] - margin)
    # end = (kpt_int[1] + margin, kpt_int[0] + margin)

    img_show = img_np.copy()[top:bottom, left:right]
    #img_show = img_np.copy()
    img_show = cv.rectangle(img_show, start, end, [0, 0, 255], 2)
    plt.imshow(img_show)
    # plt.title("original img")
    fig.axes[0].set_axis_off()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig(f"./visualizations/{counter_img}_{counter}.png")
    plt.show()
    plt.close()

    cropped = img_np[kpt_int[0] - margin: kpt_int[0] + margin, kpt_int[1] - margin: kpt_int[1] + margin]
    cropped = cv.resize(cropped, dsize=(cropped.shape[1] * scale_crop, cropped.shape[0] * scale_crop), interpolation=cv.INTER_LINEAR)
    use_cropped_only = cropped.copy()

    cross_idx = np.array([[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2], [2, -2], [1, -1], [-2, 2], [-1, 1]])

    # TODO np.round?
    yx = cropped.shape[0] // 2
    cropped[(cross_idx + yx)[:, 0], (cross_idx + yx)[:, 1]] = [0, 0, 255]
    use_cropped_mean = cropped.copy()

    use_cropped_kpts = [None] * 4
    for i, k in enumerate(kpt):
        # y, x = k[0].item(), k[1].item()
        yx = (np.round(((k).numpy() * scale_crop * scale_err)) + cropped.shape[0] // 2).astype(dtype=int)
        # y, x = yx[0], yx[1]
        cropped[(cross_idx + yx)[:, 0], (cross_idx + yx)[:, 1]] = [255, 0, 0]
        use_cropped_kpts[i] = cropped.copy()

    counter += 1
    show(use_cropped_only, "crop only")
    counter += 1
    show(use_cropped_mean, "real keypoint")

    for i in range(4):
        if i > 0:
            use_cropped_kpts_rot = np.rot90(use_cropped_kpts[i-1], i, axes=[0, 1])
            counter += 1
            show(use_cropped_kpts_rot, "rotate...")
            use_cropped_kpts_rot = np.rot90(use_cropped_kpts[i], i, axes=[0, 1])
            counter += 1
            show(use_cropped_kpts_rot, "...detect")
        title = "rotate back" if i > 0 else "detect"
        counter += 1
        show(use_cropped_kpts[i], title)

    counter += 1
    show(use_cropped_kpts[3], "'real kpt' is actually the mean")


def show_me_scale(kpt, mask, img_np, margin=11):

    counter = 0
    global counter_img
    counter_img += 1

    print(f"counter: {counter_img}")
    scale_crop = 5
    scale_err = 10
    cur_scale_err = scale_err

    def show(img, title=None):
        title=None
        img = img.copy()
        fig = plt.figure(figsize=(4, 4))
        # print(f"cur_scale_err: {cur_scale_err}")
        bound = margin * scale_crop / cur_scale_err
        # print(f"bound: {bound}")

        quarter = 0.25
        quarters = round(bound / quarter)
        while quarters > 6:
            quarter *= 2
            quarters = round(bound / quarter)

        labels = [quarter * i for i in range(-quarters, quarters + 1)]
        tickz = [margin * scale_crop / quarters * i + margin * scale_crop for i in range(-quarters, quarters + 1)]
        # print(f"labels: {labels}")
        # print(f"ticks: {tickz}")

        plt.title(title)

        # plt.xticks(tickz, labels)
        # plt.yticks(tickz, labels)
        # plt.xlabel("px")
        # plt.ylabel("px", rotation="horizontal")

        # plt.xlim(-bound, bound)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        plt.imshow(img)

        # fig.axes[0].set_axis_off()
        #plt.subplots_adjust(left=0.15, bottom=0.10, right=1.0, top=1.0, wspace=0, hspace=0)
        plt.savefig(f"./visualizations/scale_{counter_img}_{counter}.png")
        plt.show()
        plt.close()

    kpt_mean = kpt[0]
    kpt = kpt - kpt_mean
    kpt_int = torch.round(kpt_mean).to(dtype=int).tolist()
    # kpt_mean_r = kpt_mean - kpt_int

    fig = plt.figure()

    # TODO - restrict wrt. to the higher values (i.e. right and bottom)
    left = (kpt_int[1] - 50)
    right = (kpt_int[1] + 50)
    if left < 0:
        right = right - left
        left = 0
    top = (kpt_int[0] - 50)
    bottom = (kpt_int[0] + 50)
    if top < 0:
        bottom = bottom - top
        top = 0

    start = (kpt_int[1] - margin - left, kpt_int[0] - margin - top)
    end = (kpt_int[1] + margin - left, kpt_int[0] + margin - top)

    # start = (kpt_int[1] - margin, kpt_int[0] - margin)
    # end = (kpt_int[1] + margin, kpt_int[0] + margin)

    img_show = img_np.copy()[top:bottom, left:right]
    #img_show = img_np.copy()
    img_show = cv.rectangle(img_show, start, end, [0, 0, 255], 1)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(img_show)
    # plt.title("original img")
    # fig.axes[0].set_axis_off()
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig(f"./visualizations/scale_{counter_img}_{counter}.png")
    plt.show()
    plt.close()

    cropped = img_np[kpt_int[0] - margin: kpt_int[0] + margin, kpt_int[1] - margin: kpt_int[1] + margin]
    cropped = cv.resize(cropped, dsize=(cropped.shape[1] * scale_crop, cropped.shape[0] * scale_crop), interpolation=cv.INTER_LINEAR)
    use_cropped_only = cropped.copy()

    cross_idx = np.array([[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2], [2, -2], [1, -1], [-2, 2], [-1, 1]])

    # TODO np.round?
    yx = cropped.shape[0] // 2
    cropped[(cross_idx + yx)[:, 0], (cross_idx + yx)[:, 1]] = [0, 0, 255]
    use_cropped_original = cropped.copy()

    use_cropped_kpts = [None] * 10
    cur_scale_err = scale_err
    done = False
    while not done:
        cur_cropped = cropped.copy()
        for i, k in enumerate(kpt):
            if not mask[i]:
                if i == 9:
                    done = True
                continue
            # y, x = k[0].item(), k[1].item()
            yx = (np.round((k.numpy() * scale_crop * cur_scale_err)) + cropped.shape[0] // 2).astype(dtype=int)
            # y, x = yx[0], yx[1]
            ys = (cross_idx + yx)[:, 0]
            xs = (cross_idx + yx)[:, 1]
            if ys.max() >= cropped.shape[0] or ys.min() < 0 or xs.max() >= cropped.shape[1] or xs.min() < 0:
                cur_scale_err /= 1.5
                print(f"cur_scale_err decreased to {cur_scale_err}")
                break
            cur_cropped[ys, xs] = [0, 0, 255]
            use_cropped_kpts[i] = cur_cropped.copy()
            if i == 9:
                done = True

    counter += 1
    show(use_cropped_only, "crop only")
    counter += 1
    show(use_cropped_original, f"original keypoint; scale_err={cur_scale_err}")

    for i in range(10):
        if mask[i]:
            # if i > 0:
            #     use_cropped_kpts_rot = np.rot90(use_cropped_kpts[i-1], i, axes=[0, 1])
            #     counter += 1
            #     show(use_cropped_kpts_rot, "rotate...")
            #     use_cropped_kpts_rot = np.rot90(use_cropped_kpts[i], i, axes=[0, 1])
            #     counter += 1
            #     show(use_cropped_kpts_rot, "...detect")
            # title = "rotate back" if i > 0 else "detect"
            counter += 1
            show(use_cropped_kpts[i])


def rotate_experiment_loop(detector, img_to_show, err_th, show_img=True, use_mnn=False):
    img_dir = "demo_imgs/hypersim"
    files = ["{}/{}".format(img_dir, fn) for fn in os.listdir(img_dir)][:img_to_show]
    for file_path in files:
        print(f"\n\nFILE: {file_path}\n")
        for rots in range(1, 4):
            rotate_experiment(file_path, detector, rots, err_th, show_img, use_mnn)


def rotate_lowe_exp():

    def open_img(f_p):
        return np.array(Image.open(f_p))

    def read_lowe_key_file(f_n):
        with open(f_n, "r") as f:
            lines = list(f.readlines())
            kpts_c = int(lines[0].split(" ")[0])
            kpts_cv = []
            kpts_geo = []
            descs = []
            counter = 1
            for i in range(kpts_c):
                floats = [float(i) for i in lines[counter].split(" ")]
                counter += 1
                all_ints = []
                for j in range(7):
                    ints = [int(i) for i in lines[counter].split(" ") if len(i.strip()) > 0]
                    counter += 1
                    all_ints.extend(ints)
                kpts_cv.append(cv.KeyPoint(floats[1], floats[0], size=floats[2], angle=floats[3]))
                kpts_geo.append([floats[0], floats[1]])
                descs.append(np.array(all_ints, dtype=float))
        descs = np.array(descs)
        kpts_geo = np.array(kpts_geo)
        # NOTE kpts_geo thrown away
        return kpts_cv, descs

    use_mnn = True
    err_th = 10
    show_img = True
    for i in range(5):
        img_o = open_img(f"lowe/{i}_rot_3.pgm")
        kpts_0_cv, desc_0_cv = read_lowe_key_file(f"lowe/{i}_rot_3.pgm.key")
        for j in range(3):
            img_r = open_img(f"lowe/{i}_rot_{j}.pgm")
            kpts_1_cv, desc_1_cv = read_lowe_key_file(f"lowe/{i}_rot_{j}.pgm.key")
            rotations = j + 1
            rotate_exp_from_kpts(img_o, img_r, rotations, kpts_0_cv, desc_0_cv, kpts_1_cv, desc_1_cv, use_mnn, err_th, show_img)


if __name__ == "__main__":
    from superpoint_local import SuperPointDetector
    from sift_detectors import AdjustedSiftDescriptor
    detector = AdjustedSiftDescriptor(adjustment=[0., 0.])
    #detector = AdjustedSiftDescriptor(adjustment=[0.25, 0.25])
    # rotate_experiment_loop(detector, img_to_show=6, show_img=True, err_th=4, use_mnn=False)
    rotate_detail_experiment_loop(detector, img_to_show=5, show_img=True)
    scale_detail_experiment_loop(detector, img_to_show=1, show_img=True)
    # prepare_lowe(img_to_show=5)
    # rotate_lowe_exp()
