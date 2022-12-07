import cv2 as cv
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import math

import PIL
from PIL import Image

from utils import get_tentatives
# from prepare_data import scale_pil
from lowe_sift_file_descriptor import LoweSiftDescriptor
from kornia_sift import NumpyKorniaSiftDescriptor


def gcd_euclid(a, b):

    c = a % b
    if c == 0:
        return b
    else:
        return gcd_euclid(b, c)


def scale_pil(img, scale, config, mode=("LANCZOS", PIL.Image.LANCZOS), show=False):
    """
    :param img:
    :param scale: in [0, 1]
    :param show:
    :return:
    """
    real_scale = scale
    h, w = img.size
    integer_scale = config['integer_scale']
    if integer_scale:
        gcd = gcd_euclid(w, h)

        real_scale_gcd = round(gcd * scale)
        real_scale = real_scale_gcd / gcd

        fall_back = False
        if real_scale == 0.0 or math.fabs(real_scale - scale) > 0.1:
            if fall_back:
                print("WARNING: scale={} => {}".format(real_scale, scale), config)
                real_scale = scale
            else:
                raise Exception("scale {} cannot be effectively realized for w, h = {}, {} in integer domain".format(scale, w, h))

    w_sc = round(w * real_scale)
    h_sc = round(h * real_scale)
    # print(f"scaling by: {mode[0]} = {mode[1]}")
    img_r = img.resize((h_sc, w_sc), resample=mode[1])
    if show:
        show_pil(img_r)

    return img_r, real_scale


def show_pil(img):
    npa = np.array(img)
    plt.figure()
    plt.imshow(npa)
    plt.show()


class Counter:

    counter_img = 0
    counter = 0

    @staticmethod
    def reset():
        Counter.counter = 0
        Counter.counter_img = 0


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

    return mask_00, mask_10, kpts_1_geo, kpts_1_cv


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


def rotate_experiment(file_path, detector, rotations_90_deg, err_th, show_img=True, use_mnn=True, rotate90_gauss=None, rotate90_interpolation=None):
    print(f"experiment for n rotations: {rotations_90_deg}")

    old_mod_8 = False
    img_np_o = np.array(Image.open(file_path))
    if old_mod_8:
        new_h, new_w = img_np_o.shape[0] // 8 * 8, img_np_o.shape[1] // 8 * 8
        img_np_o = img_np_o[:new_h, :new_w]

    if rotate90_gauss is not None:
        detector.set_rotate_gauss(0)
    if rotate90_interpolation is not None:
        detector.set_rotate_interpolation(0)
    kpts_0_cv, desc_0_cv = detector.detectAndCompute(img_np_o, None)

    img_np_r = np.rot90(img_np_o, rotations_90_deg, [0, 1])
    if rotate90_gauss is not None and rotate90_gauss != 0:
        detector.set_rotate_gauss(rotate90_gauss)
    if rotate90_interpolation is not None and rotate90_interpolation != 0:
        detector.set_rotate_interpolation(rotate90_interpolation)
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
        # get indices of "ideal" kpts...
        indices = np.abs(kpts_0_new_geo[:, 0] - kpts_1_new_geo[:, 0] - 0.5).argsort()

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

    # TODO perform the experiments withour the clipping
    def open_img(f_p, clip=True):
        img_np_o = np.array(Image.open(f_p))
        if clip:
            new_h, new_w = img_np_o.shape[0] // 8 * 8, img_np_o.shape[1] // 8 * 8
            img_np_o = img_np_o[:new_h, :new_w]
        return img_np_o

    img_dir = "demo_imgs/hypersim"
    files = ["{}/{}".format(img_dir, fn) for fn in os.listdir(img_dir)][:img_to_show]
    for file_path in files[3:4]:
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

        for kpt in kpts_geos:
            show_me_rotate(kpt, img_np_o)


def prepare_lowe_rotation():

    def open_img(f_p):
        img_np_o = np.array(Image.open(f_p))
        img_np_o = cv.cvtColor(img_np_o, cv.COLOR_RGB2GRAY)
        return img_np_o

    # img_dir = "demo_imgs/hypersim"
    # files = ["{}/{}".format(img_dir, fn) for fn in os.listdir(img_dir)][:img_to_show]
    files = [f"demo_imgs/bark/img1.ppm"]
    for i, file_path in enumerate(files):
        img_np = open_img(file_path)
        cv.imwrite(f"demo_imgs/lowe_all/imgs/pure_rotation_{i}_rot_0.pgm", img_np)
        for j in range(3):
            img_np = np.rot90(img_np, 1, [0, 1])
            cv.imwrite(f"demo_imgs/lowe_all/imgs/pure_rotation_{i}_rot_{j + 1}.pgm", img_np)


def prepare_lowe_all():
    prepare_lowe_scaling()
    prepare_lowe_rotation()

    import shutil
    files_bark = [f"demo_imgs/bark/{f}" for f in sorted(list(os.listdir("bark"))) if f.endswith(".ppm")]
    for file in files_bark:
        img = np.array(Image.open(file))
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        cv.imwrite(f"demo_imgs/lowe_all/imgs/bark_{file[-5]}.pgm", img)
    files_boat = [f"demo_imgs/boat/{f}" for f in sorted(list(os.listdir("boat"))) if f.endswith(".pgm")]
    for file in files_boat:
        shutil.copyfile(file, f"demo_imgs/lowe_all/imgs/boat_{file[-5]}.pgm")


def prepare_lowe_scaling():

    def open_img(f_p, scale, mod_4, lanczos, resized_lin_pil):

        img = Image.open(f_p)
        h, w = img.size
        #img = np.array(img)
        if mod_4:
            right = w // 4 * 4
            bottom = h // 4 * 4
            img = img.crop((0, 0, bottom, right))
        h, w = img.size
        if scale:
            h_sc = round(h * scale)
            w_sc = round(w * scale)
            if lanczos:
                img = img.resize((h_sc, w_sc), resample=PIL.Image.LANCZOS)
                img = np.array(img)
            else:
                if resized_lin_pil:
                    img = img.resize((h_sc, w_sc), resample=PIL.Image.LINEAR)
                    img = np.array(img)
                else:
                    img = np.array(img)
                    img = cv.resize(img, dsize=(h_sc, w_sc), interpolation=cv.INTER_LINEAR)
        else:
            img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        return img
        # img_pil = Image.open(f_p)
        # if scale:
        #     img_pil, scale = scale_pil(img_pil, scale, {"integer_scale": True}, show=False)
        # img_np_r = np.array(img_pil)
        # img_np_r = cv.cvtColor(img_np_r, cv.COLOR_RGB2GRAY)
        # return img_np_r

    # img_dir = "demo_imgs/hypersim"
    # files = ["{}/{}".format(img_dir, fn) for fn in os.listdir(img_dir)][:img_to_show]
    files = [f"demo_imgs/bark/img1.ppm"]
    for lanczos in [False, True]:
        for resized_lin_pil in [False, True]:
            if lanczos and resized_lin_pil:
                continue
            for mod_4 in [False, True]:
                for i, file_path in enumerate(files):
                    res_str = "" if lanczos else f"resized_lin_pil_{str(resized_lin_pil)}"
                    file_pref = f"demo_imgs/lowe_all/imgs/pure_scale_lanczos_{str(lanczos)}_{res_str}_mod4_{str(mod_4)}_{i}"
                    img_np = open_img(file_path, None, mod_4, lanczos, resized_lin_pil)
                    cv.imwrite(f"{file_pref}_10.pgm", img_np)
                    for j in range(1, 10):
                        img_np = open_img(file_path, 0.1 * j, mod_4, lanczos, resized_lin_pil)
                        cv.imwrite(f"{file_pref}_{j}.pgm", img_np)


def scale_experiment_lowe(img_to_show, show_img=True):

    all_kpts = [np.zeros((0, 2)) for _ in range(10)]

    def open_img(f_p):
        img_np_o = np.array(Image.open(f_p))
        new_h, new_w = img_np_o.shape[0] // 8 * 8, img_np_o.shape[1] // 8 * 8
        img_np_o = img_np_o[:new_h, :new_w]
        return img_np_o

    # CONTINUE
    # for i in range(5):
    #     for scale om
    # img_dir = f"demo_imgs/lowe/imgs_scaling/{i}_rot_{j}.key"
    # FIXME
    img_dir = f"demo_imgs/lowe/imgs_scaling/foo_rot_bar.key"
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
            mask_00_and, mask_10, kpts_1_geo, _ = scale_experiment_against(kpts_0_cv, desc_0_cv, file_path, detector, scale, show_img)
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
            mask_00_and, mask_10, kpts_1_geo, _ = scale_experiment_against(kpts_0_cv, desc_0_cv, file_path, detector, scale, show_img)
            if mask_00_and.sum().item() > 0:
                mask_00_bool_and = torch.zeros(len(kpts_0_cv), dtype=bool)
                mask_00_bool_and[mask_00_and] = True
                mask_00 = mask_00 & mask_00_bool_and
                # 10 - scale_int => so that original is at 0 and then scale 0.9, 0.8, etc.
                kpts_geos[mask_00_and, 10 - scale_int, :] = kpts_1_geo[mask_10]
                kpts_geos_bool[mask_00_and, 10 - scale_int] = True

        kpts_geos_bool_sums = kpts_geos_bool.sum(dim=1)

        th = 0
        kpts_geos = kpts_geos[kpts_geos_bool_sums > th]
        kpts_geos_bool = kpts_geos_bool[kpts_geos_bool_sums > th]
        # if mask_00.sum() == 0:

        margin = 21
        for i, kpt in enumerate(kpts_geos):
            kpt_mean = kpt[0]
            kpt_int = torch.round(kpt_mean)
            show = False
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
            for mask_i in range(10):
                if mask[mask_i]:
                    all_kpts[mask_i] = np.vstack((all_kpts[mask_i], kpt[None, mask_i]))
            #all_kpts = np.vstack((all_kpts, kpt[None]))

    for i in range(10):
        m = all_kpts[i].mean(axis=0)
        v = all_kpts[i].var(axis=0)
        print(f"i: {i}/{all_kpts[i].shape[0]}: mean: {m}, variance: {v}")


def match_and_print(kpts_original, desc_original, kpts_transformed, desc_transformed, title, scale=1.0):

    ratio_threshold = 0.8
    space_dist_th = 20.0
    # space_dist_th = None
    kpts_original_rotated_geo, kpts_match_rotated_geo, _, _, tentative_matches = get_tentatives(kpts_original,
                                                                                                desc_original,
                                                                                                kpts_transformed,
                                                                                                desc_transformed,
                                                                                                ratio_threshold=ratio_threshold,
                                                                                                space_dist_th=space_dist_th)
    # mask_00 = torch.tensor([t.queryIdx for t in tentative_matches])
    # mask_10 = torch.tensor([t.trainIdx for t in tentative_matches])
    residuum = (kpts_match_rotated_geo - kpts_original_rotated_geo) * scale
    mean = residuum.mean(axis=0)
    std = np.sqrt(residuum.var(axis=0))
    print(f"{title}: mean: {mean}, std: {std}")
    ret = np.array([mean, std])
    return ret


def scale_rotate_detail_experiment_loop(detector, imgs_to_show):

    img_dir = "demo_imgs/hypersim"
    files = sorted(["{}/{}".format(img_dir, fn) for fn in os.listdir(img_dir)][:imgs_to_show])
    for mode in [("LANCSOZ", PIL.Image.LANCZOS),
                 ("NEAREST", PIL.Image.NEAREST)]:
                 # ("LINEAR", PIL.Image.LINEAR),
                 # ("NEAREST", PIL.Image.NEAREST),
                 # ("BICUBIC", PIL.Image.BICUBIC),
        scale_rotate_detail_experiment_loop_per_img(detector, files, mode)


def scale_rotate_detail_experiment_loop_per_img(detector, files, mode):

    def detect(img_np, crop):
        def modulo32(n):
            if not crop:
                return n
            n_n = n - ((n + 1) % 32)
            assert n_n % 32 == 31
            return n_n
        h, w = img_np.shape[:2]
        h = modulo32(h)
        w = modulo32(w)
        img_np = img_np[:h, :w]
        kpts, desc = detector.detectAndCompute(img_np, None)
        kpts_geo = torch.tensor([[kp.pt[1], kp.pt[0]] for kp in kpts])
        return kpts, kpts_geo, desc

    def backproject_scaled(kpts_cv, kpts_geo, scale):
        kpts_geo = kpts_geo / scale
        for i, e_kpts_cv in enumerate(kpts_cv):
            e_kpts_cv.pt = (kpts_geo[i, 1].item(), kpts_geo[i, 0].item())
        return kpts_cv, kpts_geo

    def backproject_rotated(kpts_cv, kpts_geo, img_np):
        coord0_max = img_np.shape[0] - 1
        coord1_max = img_np.shape[1] - 1
        return backproject_rotated_tuple(kpts_cv, kpts_geo, coord0_max, coord1_max)

    def backproject_rotated_tuple(kpts_cv, kpts_geo, coord0_max, coord1_max):
        rotations = 1
        for i in range(4 - rotations):
            kpts_0_new = coord1_max - kpts_geo[:, 1]
            kpts_1_new = kpts_geo[:, 0].clone()
            kpts_geo[:, 0] = kpts_0_new.clone()
            kpts_geo[:, 1] = kpts_1_new.clone()
            coord1_max, coord0_max = coord0_max, coord1_max
        for i, e_kpts_cv in enumerate(kpts_cv):
            e_kpts_cv.pt = (kpts_geo[i, 1].item(), kpts_geo[i, 0].item())
        return kpts_cv, kpts_geo

    def read_img(file_path, crop):
        img_pil = Image.open(file_path)
        w, h = img_pil.size[:2]
        def modulo32(n):
            n_n = n - ((n + 1) % 32)
            assert n_n % 32 == 31
            return n_n
        if crop:
            w = modulo32(w)
            h = modulo32(h)
            img_pil = img_pil.crop((0, 0, w, h))
            check_pil_mod(img_pil)
        return img_pil

    def check_mod(n):
        for i in range(6):
            assert n % 2 == 1
            n //= 2

    def check_pil_mod(img_pil):
        h, w = img_pil.size[:2]
        check_mod(h)
        check_mod(w)

    def check_np_mod(img_np):
        h, w = img_np.shape[:2]
        check_mod(h)
        check_mod(w)

    print(f"MODE: {mode[0]}")

    confs = [[False, False, True], [False, True, True]]
    for crop_when_read, crop_when_detect, integer_scale in confs:
        print(f"crop_when_read: {crop_when_read}")
        print(f"crop_when_detect: {crop_when_detect}")
        print(f"integer_scale: {integer_scale}")

        scales = [i / 10 for i in range(1, 10)]
        l_sc = len(scales)
        rotated_stat = np.zeros((2, 2), dtype=float)
        scaled_stat = np.zeros((l_sc, 2, 2), dtype=float)
        scaled_rotated_vs_scaled = np.zeros((l_sc, 2, 2), dtype=float)
        scaled_rotated_vs_rotated = np.zeros((l_sc, 2, 2), dtype=float)
        rotated_scaled_vs_rotated = np.zeros((l_sc, 2, 2), dtype=float)
        rotated_scaled_vs_scaled_rotated = np.zeros((l_sc, 2, 2), dtype=float)

        for file_path in files:
            print(f"FILE: {file_path}\n")

            img_pil = read_img(file_path, crop_when_read)
            img_np_original = np.array(img_pil)
            if crop_when_read:
                check_np_mod(img_np_original)
            kpts_original, kpts_original_geo, desc_original = detect(img_np_original, crop_when_detect)

            img_np_rotated = np.rot90(img_np_original, 1, [0, 1])
            if crop_when_read:
                check_np_mod(img_np_rotated)
            kpts_rotated, kpts_rotated_geo, desc_rotated = detect(img_np_rotated, crop_when_detect)
            kpts_rotated, kpts_rotated_geo = backproject_rotated(kpts_rotated, kpts_rotated_geo, img_np_rotated)
            rotated_stat += match_and_print(kpts_original, desc_original, kpts_rotated, desc_rotated, "rotated")

            img_pil_rotated = Image.fromarray(img_np_rotated)
            if crop_when_read:
                check_pil_mod(img_pil_rotated)

            for scale_int, scale in enumerate(scales):
                print(f"scale: {scale}")

                img_pil_scaled, scale_real = scale_pil(img_pil, scale, {"integer_scale": integer_scale}, mode=mode, show=False)
                # print(f"scale_real: {scale_real}")
                img_np_scaled = np.array(img_pil_scaled)
                kpts_scaled, kpts_scaled_geo, desc_scaled = detect(img_np_scaled, crop_when_detect)
                kpts_scaled, kpts_scaled_geo = backproject_scaled(kpts_scaled, kpts_scaled_geo, scale_real)
                scaled_stat[scale_int] += match_and_print(kpts_original, desc_original, kpts_scaled, desc_scaled, "scaled", scale=scale_real)

                # kpts_scaled, kpts_scaled_geo, desc_scaled = detect(img_np_scaled)
                img_np_scaled_rotated = np.rot90(img_np_scaled, 1, [0, 1])
                kpts_scaled_rotated, kpts_scaled_rotated_geo, desc_scaled_rotated = detect(img_np_scaled_rotated, crop_when_detect)
                kpts_scaled_rotated, kpts_scaled_rotated_geo = backproject_rotated(kpts_scaled_rotated, kpts_scaled_rotated_geo, img_np_scaled_rotated)
                kpts_scaled_rotated, kpts_scaled_rotated_geo = backproject_scaled(kpts_scaled_rotated, kpts_scaled_rotated_geo, scale_real)
                scaled_rotated_vs_scaled[scale_int] += match_and_print(kpts_scaled, desc_scaled, kpts_scaled_rotated, desc_scaled_rotated, "scaled and rotated vs scaled", scale_real)
                scaled_rotated_vs_rotated[scale_int] += match_and_print(kpts_rotated, desc_rotated, kpts_scaled_rotated, desc_scaled_rotated, "scaled and rotated vs rotated", scale_real)

                img_pil_rotated_scaled, scale_real2 = scale_pil(img_pil_rotated, scale, {"integer_scale": integer_scale}, mode=mode, show=False)
                # print(f"scale_real2: {scale_real2}")
                assert scale_real2 == scale_real
                img_np_rotated_scaled = np.array(img_pil_rotated_scaled)
                kpts_rotated_scaled, kpts_rotated_scaled_geo, desc_rotated_scaled = detect(img_np_rotated_scaled, crop_when_detect)
                kpts_rotated_scaled, kpts_rotated_scaled_geo = backproject_scaled(kpts_rotated_scaled, kpts_rotated_scaled_geo, scale_real2)
                kpts_rotated_scaled, kpts_rotated_scaled_geo = backproject_rotated(kpts_rotated_scaled, kpts_rotated_scaled_geo, img_np_rotated)

                rotated_scaled_vs_rotated[scale_int] += match_and_print(kpts_rotated, desc_rotated, kpts_rotated_scaled, desc_rotated_scaled, "rotated and scaled vs rotated", scale_real)
                rotated_scaled_vs_scaled_rotated[scale_int] += match_and_print(kpts_scaled_rotated, desc_scaled_rotated, kpts_rotated_scaled, desc_rotated_scaled, "rotated and scaled vs scaled rotated", scale_real)

                fig, axs = plt.subplots(1, 2, figsize=(10, 7))
                fig.tight_layout()
                fig.suptitle("foo")
                axs[0].set_axis_off()
                axs[0].set_title(f"original: {file_path.split('/')[-1]}")
                axs[0].imshow(img_np_original)
                axs[1].set_axis_off()
                axs[1].set_title(f"'rotated_scaled' - 'scaled_rotated': {mode[0]} at {scale}")
                axs[1].imshow(img_np_rotated_scaled - img_np_scaled_rotated)
                plt.savefig(f"demo_imgs/scale_rotate/diff_{mode[0]}_{scale}_{file_path.split('/')[-1]}.png")
                # plt.show()
                plt.close()

        print(f"MODE: {mode[0]}")
        print(f"crop_when_read: {crop_when_read}")
        print(f"crop_when_detect: {crop_when_detect}")
        print(f"integer_scale: {integer_scale}")

        l = len(files)
        rotated_stat /= l
        print(f"rotated: mean: {rotated_stat[0]}, std: {rotated_stat[1]}")

        def print_stat(stat, name):
            stat /= l
            for scale in range(9):
                print(f"{name}: mean: {stat[scale, 0]}, std: {stat[scale, 1]}")

        print_stat(scaled_stat, "scaled_stat")
        print_stat(scaled_rotated_vs_scaled, "scaled_rotated_vs_scaled")
        print_stat(scaled_rotated_vs_rotated, "scaled_rotated_vs_rotated")
        print_stat(rotated_scaled_vs_rotated, "rotated_scaled_vs_rotated")
        print_stat(rotated_scaled_vs_scaled_rotated, "rotated_scaled_vs_scaled_rotated")


def show_me_rotate(kpt, img_np, margin=11):

    kpt_mean = kpt.mean(dim=0)
    kpt_int = torch.round(kpt_mean).to(int)
    if kpt_int[0].item() < margin or kpt_int[0].item() > img_np.shape[0] - 1 - margin:
        return
    if kpt_int[1].item() < margin or kpt_int[1].item() > img_np.shape[1] - 1 - margin:
        return

    # counter = 0
    # global counter_img
    Counter.counter_img += 1
    # print(f"counter: {counter_img}")
    # scale_crop = 5
    # scale_err = 1

    def show(img, title=None):
        title = None
        Counter.counter += 1
        img = img.copy()
        fig = plt.figure(figsize=(4, 4))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # bound = margin / scale_err
        #
        # quarters = round(bound / 0.25)
        # labels = [0.25 * i for i in range(-quarters, quarters + 1)]
        # tickz = [12.5 * i + 55 for i in range(-quarters, quarters + 1)]
        #
        # plt.xticks(tickz, labels) #, rotation='vertical')
        # plt.yticks(tickz, labels)
        # plt.xlabel("px")
        # plt.ylabel("px", rotation="horizontal")

        # plt.xlim(-bound, bound)
        plt.imshow(img)
        # plt.title(title)
        # plt.subplots_adjust(left=0.15, bottom=-0.2, right=1.0, top=1.4, wspace=200, hspace=100)
        # plt.subplots_adjust(left=0.15, bottom=0.10, right=1.0, top=1.0, wspace=0, hspace=0)
        plt.savefig(f"./visualizations/rotation_{Counter.counter_img}_{Counter.counter}.png")
        plt.show()
        plt.close()

    def perform_crop(img_np_l, center, margin_b, margin_small, scale_crop_l, crop=True):

        left = (center[1] - margin_b)
        right = (center[1] + margin_b + 1)
        if left < 0:
            right = right - left
            left = 0
        top = (center[0] - margin_b)
        bottom = (center[0] + margin_b + 1)
        if top < 0:
            bottom = bottom - top
            top = 0

        img_show = img_np_l.copy()[top:bottom, left:right]

        center[0] -= top
        center[1] -= left
        rect = np.array(
            [[center[0] - margin_small, center[1] - margin_small],
             [center[0] + margin_small + 1, center[1] + margin_small + 1]])
        if crop:
            cropped = img_show[rect[0, 0]:rect[1, 0], rect[0, 1]:rect[1, 1]]
        # fig = plt.figure()
        # plt.imshow(cropped)
        # plt.title("cropped")
        # # fig.axes[0].set_axis_off()
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        # plt.show()
        # plt.close()
        # start = (rect[0, 1], rect[0, 0])
        # end = (rect[1, 1], rect[1, 0])
        # img_show = cv.rectangle(img_show, start, end, [0, 0, 255], 1)
        #
        # fig = plt.figure()
        # plt.imshow(img_show)
        # plt.title("cropp 2")
        # # fig.axes[0].set_axis_off()
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        # plt.savefig(f"./visualizations/{counter_img}_{counter}.png")
        # plt.show()
        # plt.close()

        #img_show = img_np.copy()
        #img_show = cv.rectangle(img_show, start, end, [0, 0, 255], 1)
        img_show = cv.resize(img_show, dsize=(int(img_show.shape[1] * scale_crop_l), int(img_show.shape[0] * scale_crop_l)), interpolation=cv.INTER_NEAREST)
        # start_l = (center_cv[1] - margin_small - left, center_cv[0] - margin_small - top)
        # end_l = (center_cv[1] + margin_small - left, center_cv[0] + margin_small - top)
        # start = (center_cv[1] - margin_small * scale_crop - left, center_cv[0] - margin_small * scale_crop - top)
        # end = (center_cv[1] + margin_small * scale_crop - left, center_cv[0] + margin_small * scale_crop - top)
        rect = (rect * scale_crop_l).astype(dtype=int)
        start = (rect[0, 1], rect[0, 0])
        end = (rect[1, 1], rect[1, 0])
        if crop:
            img_show = cv.rectangle(img_show, start, end, [0, 0, 255], 4)

        Counter.counter += 1
        img_show_o = img_show.copy()
        for i in range(4):
            if i > 0:
                img_show = np.rot90(img_show, 1, [0, 1])
            fig = plt.figure(figsize=(4, 4))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.imshow(img_show)
            # plt.title("original img")
            #fig.axes[0].set_axis_off()
            #plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            if i == 0:
                name = f"./visualizations/rotation_{Counter.counter_img}_{Counter.counter}.png"
            else:
                name = f"./visualizations/rotation_{Counter.counter_img}_{Counter.counter}_{i}.png"
            plt.savefig(name)
            plt.show()
            plt.close()

        # cropped = img_np[kpt_int_l[0] - margin_small: kpt_int_l[0] + margin_small, kpt_int_l[1] - margin_small: kpt_int_l[1] + margin_small]
        #cropped = cv.resize(cropped, dsize=(cropped.shape[1] * scale_crop, cropped.shape[0] * scale_crop), interpolation=cv.INTER_NEAREST)
        if crop:
            return cropped
        else:
            return img_show_o

    scale_crop_l = 1
    scale_crop_l *= 10
    cropped = perform_crop(img_np, kpt_int.tolist(), 50, 11, scale_crop_l)
    use_cropped_only = cropped.copy()
    scale_crop_l *= 2
    cropped = perform_crop(cropped, [cropped.shape[0] // 2,] * 2, 11, 6, scale_crop_l)
    scale_crop_l *= 2
    cropped = perform_crop(cropped, [cropped.shape[0] // 2,] * 2, 6, 1, scale_crop_l)
    # counter += 1
    # scale_crop_l *= 2
    # cropped = perform_crop(cropped, [cropped.shape[0] // 2,] * 2, 10, 5, scale_crop_l)
    # counter += 1
    # scale_crop_l *= 2
    # cropped = perform_crop(cropped, [cropped.shape[0] // 2,] * 2, 5, 1, scale_crop_l)
    scale_crop_l *= 1
    cropped = perform_crop(cropped, [cropped.shape[0] // 2,] * 2, 1, 0, scale_crop_l, crop=False)

    contrast = 6
    cropped_mean = cropped.mean(axis=0).mean(axis=0)
    cropped = (cropped_mean + (cropped - cropped_mean) * contrast).astype(int)
    show(cropped)

    # scale_crop_l *= 3
    # counter += 1
    # cropped = perform_crop(cropped, [cropped.shape[0] // 2,] * 2, 5, 1, scale_crop_l)
    #use_cropped_only = cropped.copy()

    #cross_idx = np.array([[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2], [2, -2], [1, -1], [-2, 2], [-1, 1]])
    cross_idx = np.array([[-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0], [0, -2], [0, -1], [0, 2], [0, 1]])

    kpt_mean = kpt.mean(dim=0)
    # kpt = kpt - kpt_mean
    kpt_mean_int = torch.round(kpt_mean).to(dtype=int)
    kpt_mean_r = kpt_mean - kpt_mean_int

    scale_f = cropped.shape[0] / 3
    yx = (torch.tensor(cropped.shape[:2]) / 2) + kpt_mean_r * scale_f
    yx = yx.to(int).numpy()
    cropped[(cross_idx + yx)[:, 0], (cross_idx + yx)[:, 1]] = [0, 0, 255]
    use_cropped_mean = cropped.copy()

    use_cropped_kpts = [None] * 4
    for i, k in enumerate(kpt):
        # y, x = k[0].item(), k[1].item()
        # yx = (np.round(((k).numpy() * scale_crop * scale_err)) + cropped.shape[0] // 2).astype(dtype=int)
        yx = (k - kpt_mean_int) * scale_f + torch.tensor(cropped.shape[:2]) / 2
        yx = yx.to(int).numpy()
        # y, x = yx[0], yx[1]
        cropped[(cross_idx + yx)[:, 0], (cross_idx + yx)[:, 1]] = [255, 0, 0]
        use_cropped_kpts[i] = cropped.copy()

    show(use_cropped_only, "crop only")
    show(use_cropped_mean, "real keypoint")

    for i in range(4):
        if i > 0:
            use_cropped_kpts_rot = np.rot90(use_cropped_kpts[i-1], i, axes=[0, 1])
            show(use_cropped_kpts_rot, "rotate...")
            use_cropped_kpts_rot = np.rot90(use_cropped_kpts[i], i, axes=[0, 1])
            show(use_cropped_kpts_rot, "...detect")
        title = "rotate back" if i > 0 else "detect"
        show(use_cropped_kpts[i], title)

    show(use_cropped_kpts[3], "'real kpt' is actually the mean")


def show_me_scale(kpt, mask, img_np, margin=11):

    Counter.counter = 0
    Counter.counter_img += 1

    print(f"counter: {Counter.counter_img}")
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
        plt.savefig(f"./visualizations/scale_{Counter.counter_img}_{Counter.counter}.png")
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
    plt.savefig(f"./visualizations/scale_{Counter.counter_img}_{Counter.counter}.png")
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

    Counter.counter += 1
    show(use_cropped_only, "crop only")
    Counter.counter += 1
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
            Counter.counter += 1
            show(use_cropped_kpts[i])


def rotate_experiment_loop(detector, img_to_show, err_th, show_img=True, use_mnn=False, compensate_gauss=None, compensate_interpolation=None):
    img_dir = "demo_imgs/hypersim"
    files = ["{}/{}".format(img_dir, fn) for fn in os.listdir(img_dir)][:img_to_show]
    for file_path in files:
        print(f"\n\nFILE: {file_path}\n")
        for rots in range(1, 4):
            rotate90_gauss = None
            if compensate_gauss is not None:
                rotate90_gauss = 4 - rots if compensate_gauss else 0
            rotate90_interpolation = None
            if compensate_interpolation is not None:
                rotate90_interpolation = 4 - rots if compensate_interpolation else 0
            rotate_experiment(file_path, detector, rots, err_th, show_img, use_mnn, rotate90_gauss, rotate90_interpolation)


def rotate_lowe_exp():

    def open_img(f_p):
        return np.array(Image.open(f_p))

    desc = LoweSiftDescriptor()
    use_mnn = True
    err_th = 10
    show_img = True
    for i in range(5):
        img_o = open_img(f"demo_imgs/lowe/img_rotation/{i}_rot_3.pgm")
        kpts_0_cv, desc_0_cv = desc.detectAndCompute(f"demo_imgs/lowe/{i}_rot_3.pgm.key")
        for j in range(3):
            img_r = open_img(f"demo_imgs/lowe/img_rotation/{i}_rot_{j}.pgm")
            kpts_1_cv, desc_1_cv = desc.detectAndCompute(f"demo_imgs/lowe/{i}_rot_{j}.pgm.key")
            rotations = j + 1
            rotate_exp_from_kpts(img_o, img_r, rotations, kpts_0_cv, desc_0_cv, kpts_1_cv, desc_1_cv, use_mnn, err_th, show_img)


if __name__ == "__main__":
    # from superpoint_local import SuperPointDetector
    from sift_detectors import AdjustedSiftDescriptor
    detector = AdjustedSiftDescriptor(adjustment=[0., 0.])

    scale_rotate_detail_experiment_loop(detector, imgs_to_show=5)

    # detector = NumpyKorniaSiftDescriptor()
    #detector = AdjustedSiftDescriptor(adjustment=[0.25, 0.25])
    #rotate_experiment_loop(detector, img_to_show=1, show_img=True, err_th=4, use_mnn=False)
    # rotate_detail_experiment_loop(detector, img_to_show=5, show_img=True)
    # scale_detail_experiment_loop(detector, img_to_show=1, show_img=True)
    # prepare_lowe(img_to_show=5)
    # prepare_lowe_scaling(img_to_show=5)
    # rotate_lowe_exp()
    # prepare_lowe_all()
