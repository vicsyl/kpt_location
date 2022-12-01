import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import kornia.utils as KU
from kornia.geometry.transform import ScalePyramid
from utils import show_torch


def mnn(kpts, kpts_scales, kpts_r, kpts_r_scales, scale, config):

    err_th = config['err_th']
    err_th = err_th / scale
    scale_ratio_th = config['scale_ratio_th']
    half_pixel_adjusted = config['half_pixel_adjusted']
    #kpts_reprojected_or = kpts_r / scale
    if half_pixel_adjusted:
        magic_scale = scale / 2 # hypothesis
        adjustment = torch.ones(2) * magic_scale
        kpts_r = kpts_r + adjustment

    kpts_reprojected = kpts_r / scale

    d_mat = torch.cdist(kpts, kpts_reprojected)

    # min0 => minima of resized
    min0 = torch.min(d_mat, dim=0)
    k = 3
    if k > d_mat.shape[0]:
        up_to_k_min = torch.zeros((k, 0))
    else:
    # FIXME dmat.shape[0] can be less than k! -> I will still need k values (hstack later on)
        up_to_k_min = torch.topk(d_mat, k, largest=False, axis=0).values # [k, dim]
    min1 = torch.min(d_mat, dim=1)

    mask = min1[1][min0[1]] == torch.arange(0, min0[1].shape[0])
    mask_th = mask & (min0[0] < err_th)

    verify = True
    if verify:
        for i in range(min0[1].shape[0]):
            if mask_th[i]:
                assert min1[1][min0[1][i]] == i
                assert min0[0][i] < err_th

    kpts0 = kpts[min0[1][mask_th]]
    kpts_scales = kpts_scales[min0[1][mask_th]]

    kpts1 = kpts_r[mask_th]
    kpts_r_scales = kpts_r_scales[mask_th]

    dists = min0[0][mask_th]
    diffs = (kpts_reprojected[mask_th] - kpts0) * scale

    if verify:
        ds = torch.diag(torch.cdist(kpts0, kpts_reprojected[mask_th]))
        # FIXME
        #assert torch.allclose(ds, dists)

    # now filter based on the scale ratio threshold
    kpts_r_scales_backprojected = kpts_r_scales / scale
    scale_ratios = kpts_r_scales_backprojected / kpts_scales
    if scale_ratio_th is not None:
        mask_ratio_th = 1 + torch.abs(1 - scale_ratios) < scale_ratio_th
        diffs = diffs[mask_ratio_th]
        kpts0 = kpts0[mask_ratio_th]
        kpts1 = kpts1[mask_ratio_th]
        kpts_scales = kpts_scales[mask_ratio_th]
        kpts_r_scales = kpts_r_scales[mask_ratio_th]
        scale_ratios = scale_ratios[mask_ratio_th]

    min_distances_reprojected = min0[0] * scale
    up_to_k_min = up_to_k_min * scale
    if up_to_k_min.shape[1] > 0:
        assert torch.all(up_to_k_min[0, :] == min_distances_reprojected)
    # distances for reprojected
    return kpts0, kpts_scales, kpts1, kpts_r_scales, diffs, scale_ratios, up_to_k_min


def show_pil(img):
    npa = np.array(img)
    plt.figure()
    plt.imshow(npa)
    plt.show()


def gcd_euclid(a, b):

    c = a % b
    if c == 0:
        return b
    else:
        return gcd_euclid(b, c)


def crop_patches(img_t, kpts, margins, heatmap_t):
    def slice_np(np_data):
        return [np_data[kp_i[0] - margins[i]: kp_i[0] + margins[i] + 1,
         kp_i[1] - margins[i]: kp_i[1] + margins[i] + 1] for i, kp_i in enumerate(kpts)]

    patches = slice_np(img_t)
    if heatmap_t is not None:
        patches_hm = slice_np(heatmap_t)
        return [patches, patches_hm]
    else:
        return [patches]


def compare_patches(patches0, patches1, diffs):

    cols = 8
    cols = min(cols, max(2, len(patches0)))

    fig, axs = plt.subplots(2, cols, figsize=(12, 4))
    fig.suptitle("Patches comparison")

    for n in range(min(cols, len(patches0))):
        for r in range(2):
            if r == 0:
                title = "err=\n{:.02f},{:.02f}".format(diffs[n, 0].item(), diffs[n, 1].item())
                axs[r, n].set_title(title)
            axs[r, n].set_axis_off()
            patch_o = patches0[n] if r == 0 else patches1[n]
            patch = patch_o.clone().detach()
            axs[r, n].imshow(patch.numpy())
    plt.show()


def scale_pil(img, scale, config, show=False):
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
                # wand_log_me("WARNING: scale={} => {}".format(real_scale, scale), config)
                real_scale = scale
            else:
                raise Exception("scale {} cannot be effectively realized for w, h = {}, {} in integer domain".format(scale, w, h))

    w_sc = round(w * real_scale)
    h_sc = round(h * real_scale)
    img_r = img.resize((h_sc, w_sc), resample=PIL.Image.LANCZOS)
    if show:
        show_pil(img_r)

    return img_r, real_scale


def get_pil_img(path, show=False):

    img = Image.open(path)
    if show:
        show_pil(img)
    #log_me("original size: {}".format(img.size))
    return img


def possibly_refl_image(config, img):
    refl = config.get('reflection', None)
    if refl:
        refl = refl.upper()
    if refl == "XY":
        img = np.flip(img, 0).copy()
        img = np.flip(img, 1).copy()
    elif refl == "Y":
        img = np.flip(img, 0).copy()
    elif refl == "X":
        img = np.flip(img, 1).copy()
    elif not refl:
        pass
    else:
        raise Exception("unknown value for reflection: {}".format(refl))
    return img


def possibly_to_grey_scale(config, img):
    to_grey_scale = config['to_grey_scale']
    if to_grey_scale:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    return img


def pil_img_transforms(img, config):
    img = np.array(img)
    img = possibly_to_grey_scale(config, img)
    img = possibly_refl_image(config, img)
    # FIXME a bit of a hack
    # FIXME temporary fix (use is hm_relevant)
    if config['detector'].lower().__contains__("superpoint"):
        new_h = (img.shape[0] // 8) * 8
        new_w = (img.shape[1] // 8) * 8
        img = img[:new_h, :new_w]
    return img


def get_img_tuple(path, scale, config, show=False):

    img = get_pil_img(path, show)
    img_r, real_scale = scale_pil(img, scale=scale, config=config, show=show)
    img = pil_img_transforms(img, config)
    img_r = pil_img_transforms(img_r, config)

    return img, img_r, real_scale


def write_patch(patch, dr, file_name_prefix, out_map, out_dir, ds_config, augment_index):

    write_imgs = ds_config['write_imgs']
    dr.augmented = "original" if augment_index == 0 else "augmented"
    file_name = "{}_{}.png".format(file_name_prefix, augment_index)
    out_map["metadata"][file_name] = dr
    img_out_path = "{}/data/{}".format(out_dir, file_name)
    if write_imgs:
        cv.imwrite(img_out_path, patch.numpy())


def get_pyr_from_path(path, config):
    img = Image.open(path)
    img_np = pil_img_transforms(img, config)
    return get_pyr_from_np(img_np)


def get_pyr_from_np(img_np):
    img_t3 = KU.image_to_tensor(img_np, False).float() / 255.
    return get_pyr_from_torch(img_t3)


def get_pyr_from_torch(img_t):
    scale_pyr = ScalePyramid(3, 1.6, 32, double_image=True)
    scale_pyr, sigmas, pixel_dists = scale_pyr(img_t)
    return scale_pyr, sigmas, pixel_dists


def show_pyr(file_path, ds_config):
        pyr, sigmas, pixel_dists = get_pyr_from_path(file_path, ds_config)
        pyr = [[level[0, :, i:i+1] for i in range(6)] for level in pyr]
        for i, level in enumerate(pyr):
            for j, i_pyr in enumerate(level):
                show_torch(i_pyr, title=f"pyr level {i + 1} sigma ix {j + 1}, sigma value: {sigmas[i][0, j]}, pd: {pixel_dists[i][0, j]}")
