import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import PIL
import torch
import os
import math
from config import *


# FIXME: use proper logging
def log_me(s):
    print(s)


def mnn(kpts, kpts_scales, kpts_r, kpts_r_scales, scale, th):

    kpts_reprojected = kpts_r / scale
    d_mat = torch.cdist(kpts, kpts_reprojected)
    min0 = torch.min(d_mat, dim=0)
    min1 = torch.min(d_mat, dim=1)

    mask = min1[1][min0[1]] == torch.arange(0, min0[1].shape[0])

    mask_th = mask & (min0[0] < th)

    verify = True
    if verify:
        for i in range(min0[1].shape[0]):
            if mask_th[i]:
                assert min1[1][min0[1][i]] == i
                assert min0[0][i] < th

    kpts0 = kpts[min0[1][mask_th]]
    kpts_scales = kpts_scales[min0[1][mask_th]]

    kpts1 = kpts_r[mask_th]
    kpts_r_scales = kpts_r_scales[mask_th]

    dists = min0[0][mask_th]
    diffs = (kpts0 - kpts_reprojected[mask_th]) * scale

    if verify:
        ds = torch.diag(torch.cdist(kpts0, kpts_reprojected[mask_th]))
        assert torch.allclose(ds, dists)

    return kpts0, kpts_scales, kpts1, kpts_r_scales, diffs


def scale_pil(img, scale, show=False):
    """
    :param img:
    :param scale: in [0, 1]
    :param show:
    :return:
    """
    h, w = img.size
    gcd = gcd_euclid(w, h)

    real_scale_gcd = round(gcd * scale)
    real_scale = real_scale_gcd / gcd

    fall_back = True
    if real_scale == 0.0 or math.fabs(real_scale - scale) > 0.1:
        if fall_back:
            log_me("WARNING: scale={} => {}".format(real_scale, scale))
            real_scale = scale
        else:
            raise Exception("scale {} cannot be effectively realized for w, h = {}, {} in integer domain".format(scale, w, h))

    w_sc = int(w * real_scale)
    h_sc = int(h * real_scale)
    img_r = img.resize((h_sc, w_sc), resample=PIL.Image.LANCZOS)
    log_me("scaled to: {}".format(img_r.size))
    if show:
        show_pil(img_r)

    print("real scale: {}".format(real_scale))
    return img_r, real_scale


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


def get_default_detector():
    detector = cv.SIFT_create()
    return detector


def detect_kpts(img_pil, scale_th, const_patch_size, detector=get_default_detector(), show=False):

    if const_patch_size is not None:
            assert const_patch_size % 2 == 1, "doesn't work that way"

    npa = np.array(img_pil)
    h, w, c = npa.shape

    kpts = detector.detect(npa, mask=None)
    if len(kpts) == 0:
        return [], []

    kpt_f = np.array([[kp.pt[1], kp.pt[0]] for kp in kpts])
    kpt_i = np.round(kpt_f).astype(int)

    scales = np.array([kp.size for kp in kpts])
    # NOTE in original image it just means it's not detected on the edge
    # even though the patch will not be put into the ds, which is still reasonable
    if const_patch_size is not None:
        margin = np.ones(scales.shape[0], dtype=int) * const_patch_size // 2
    else:
        margin = np.ceil(scales / 2).astype(int)

    mask = scales > scale_th
    mask = mask & (kpt_i[:, 0] >= margin) & (kpt_i[:, 1] >= margin)
    mask = mask & (kpt_i[:, 0] < h - margin) & (kpt_i[:, 1] < w - margin)
    kpt_f = kpt_f[mask]
    scales = scales[mask]

    if show:
        npac = npa.copy()
        cv.drawKeypoints(npa, kpts, npac, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure()
        plt.imshow(npac)
        plt.show()

    return torch.from_numpy(kpt_f), torch.from_numpy(scales)


def print_and_check_margins(kpt_i, margins_np, img_t):
    print()
    for i, kp_i in list(enumerate(kpt_i)):
        t = ((kp_i[0] - margins_np[i]).item(),
              (kp_i[0] + margins_np[i]).item() + 1,
              (kp_i[1] - margins_np[i]).item(),
              (kp_i[1] + margins_np[i]).item() + 1)
        print(t, img_t.shape)


def show_patches(patches, label, detect):

    cols = 5
    rows = 5

    if detect:
        detector = get_default_detector()

    fig, axs = plt.subplots(rows, cols, figsize=(5, 5))
    fig.suptitle(label)

    for ix in range(rows):
        for iy in range(cols):
            axs[ix, iy].set_axis_off()
            if ix * cols + iy >= len(patches):
                axs[ix, iy].imshow(np.ones((32, 32, 3), np.uint8) * 255)
                continue
            patch_to_show = patches[ix * cols + iy].numpy().copy()
            if detect:
                kpts = detector.detect(patch_to_show, mask=None)
                cv.drawKeypoints(patch_to_show, kpts, patch_to_show, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            axs[ix, iy].imshow(patch_to_show)
    plt.show()
    plt.close()


def slice_patches(img, kpts, margins):
    patches = [img[kp_i[0] - margins[i]: kp_i[0] + margins[i] + 1,
               kp_i[1] - margins[i]: kp_i[1] + margins[i] + 1][None] for i, kp_i in enumerate(kpts)]
    return [p[0] for p in patches]


def get_patches(img_pil, kpt_f, kpt_scales, const_patch_size, show_few=False):
    """
    :param img_pil:
    :param kpt_f:
    :param kpt_scales:
    :param scale_th:
    :param show_few:
    :return: patches: List[Tensor]
    """

    img_n = np.array(img_pil)
    img_t = torch.tensor(img_n)

    kpt_i = torch.round(kpt_f).to(torch.int)
    if const_patch_size is not None:
        assert const_patch_size % 2 == 1, "doesn't work that way"
        margins_np = np.ones(kpt_scales.shape[0], dtype=int) * const_patch_size // 2
    else:
        margins = torch.ceil(kpt_scales / 2.0)
        margins_np = margins.to(torch.int).numpy()
    # print_and_check_margins(kpt_i, margins_np, img_t)

    patches = slice_patches(img_t, kpt_i, margins_np)

    if show_few:

        detector = get_default_detector()
        kpts = detector.detect(img_n, mask=None)
        npac = img_n.copy()
        cv.drawKeypoints(img_n, kpts, npac, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_tc = torch.tensor(npac)
        ps_kpts = slice_patches(img_tc, kpt_i, margins_np)

        show_patches(ps_kpts, "Patches - original", detect=False)
        show_patches(patches, "Patches - redetected", detect=True)

    return patches


def compare_patches(patches0, patches1, diffs):

    cols = 8
    cols = min(cols, len(patches0))

    fig, axs = plt.subplots(2, cols, figsize=(12, 4))
    fig.suptitle("Patches comparison")

    for n in range(cols):
        for r in range(2):
            if r == 0:
                axs[r, n].set_title("err=\n{:.02f},{:.02f}".format(diffs[n, 0].item(), diffs[n, 1].item()))
            axs[r, n].set_axis_off()
            patch_o = patches0[n] if r == 0 else patches1[n]
            patch = patch_o.clone().detach()
            axs[r, n].imshow(patch.numpy())
    plt.show()


def get_img_tuple(path, scale, show=False):

    img = Image.open(path)
    if show:
        show_pil(img)
    log_me("original size: {}".format(img.size))

    img_r, real_scale = scale_pil(img, scale=scale, show=show)
    return img, img_r, real_scale


def process_patches_for_file(file_path,
                             config,
                             out_map,
                             key="",
                             compare=True,
                             show=False):
    scale = config['down_scale']
    err_th = config['err_th']
    out_dir = get_full_ds_dir(config)
    min_scale_th = config['min_scale_th']
    const_patch_size = config.get('const_patch_size')

    print("Processing: {}".format(file_path))

    # convert and show the image
    img, img_r, real_scale = get_img_tuple(file_path, scale)

    kpts, scales = detect_kpts(img, min_scale_th, const_patch_size, show=show)
    kpts_r, scales_r = detect_kpts(img_r, min_scale_th*real_scale, const_patch_size, show=show)

    if len(kpts) == 0 or len(kpts_r) == 0:
        return

    kpts, scales, kpts_r, scales_r, diffs = mnn(kpts, scales, kpts_r, scales_r, real_scale, err_th)

    patches = get_patches(img, kpts, scales, const_patch_size, show_few=show)
    patches_r = get_patches(img_r, kpts_r, scales_r, const_patch_size, show_few=show)

    if compare:
        compare_patches(patches, patches_r, diffs)

    file_name_prefix = "{}_{}".format(key, file_path[file_path.rfind("/") + 1:file_path.rfind(".")])
    for i in range(len(patches)):
        patch = patches_r[i]
        value = (*diffs[i], patch.shape[0])
        file_name = "{}_{}.png".format(file_name_prefix, i)
        out_map[file_name] = (value[0].item(), value[1].item(), value[2])
        img_out_path = "{}/{}".format(out_dir, file_name)
        cv.imwrite(img_out_path, patch.numpy())


def prepare_data(config, in_dirs, keys):

    ends_with = config['ends_with']
    max_items = config['max_items']
    const_patch_size = config.get('const_patch_size')
    if const_patch_size is not None:
            assert const_patch_size % 2 == 1, "doesn't work that way"

    out_dir = get_full_ds_dir(config)
    clean = config['clean_out_dir']

    if clean:
        try:
            path = '{}/a_values.txt'.format(out_dir)
            os.remove(path)
        except:
            print("couldn't remove {}".format(path))
        files = glob.glob('{}/*.png'.format(out_dir))
        for path in files:
            try:
                os.remove(path)
            except:
                print("couldn't remove {}".format(path))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_map = {}
    all = max_items

    for i, in_dir in enumerate(in_dirs):

        counter = 0

        if not max_items:
            all = len([fn for fn in os.listdir(in_dir) if fn.endswith(ends_with)])

        print("Processing {}".format(in_dir))
        key = keys[i]

        for file_name in os.listdir(in_dir):

            if not file_name.endswith(ends_with):
                continue
            counter += 1
            if max_items and counter > max_items:
                break

            path = "{}/{}".format(in_dir, file_name)
            print("{}/{}".format(counter, all))
            process_patches_for_file(file_path=path,
                                     config=config,
                                     out_map=out_map,
                                     key=key,
                                     compare=False,
                                     show=False)

    err = 0.0
    for fn in out_map:
        err_entry = torch.tensor(out_map[fn][:2])
        err += (err_entry @ err_entry.T).item()
    err = err / len(out_map)

    with open("{}/a_values.txt".format(out_dir), "w") as md_file:
        md_file.write("# entries: {}\n".format(len(out_map)))
        md_file.write("# detector default mean error: {}\n".format(err))
        for (fn, value) in out_map.items():
            to_write = "{}, {}, {}, {}\n".format(fn, value[0], value[1], value[2])
            md_file.write(to_write)


if __name__ == "__main__":

    config = get_config()
    in_dirs = config['in_dirs']
    keys = config['keys']

    prepare_data(config, in_dirs, keys)
