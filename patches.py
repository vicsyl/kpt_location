import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import PIL
import torch
import os
import math


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
    img_r = img.resize((h_sc, w_sc), resample=PIL.Image.Resampling.LANCZOS)
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


def detect(img_pil, scale_th, detector=get_default_detector(), show=False):

    npa = np.array(img_pil)
    h, w, c = npa.shape

    kpts = detector.detect(npa, mask=None)

    kpt_f = np.array([[kp.pt[1], kp.pt[0]] for kp in kpts])
    kpt_i = np.round(kpt_f).astype(int)

    scales = np.array([kp.size for kp in kpts])
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


def get_patches(img_pil, kpt_f, kpt_scales, scale_th, show_few=False):
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
                             out_dir,
                             out_dict,
                             scale,
                             err_th,
                             compare=True,
                             show=False):

    print("Processing: {}".format(file_path))

    # convert and show the image
    img, img_r, real_scale = get_img_tuple(file_path, scale)

    min_scale_th = 15.0
    kpts, scales = detect(img, min_scale_th, show=show)
    kpts_r, scales_r = detect(img_r, min_scale_th*real_scale, show=show)

    if len(kpts) == 0 or len(kpts_r) == 0:
        return

    kpts, scales, kpts_r, scales_r, diffs = mnn(kpts, scales, kpts_r, scales_r, real_scale, err_th)

    patches = get_patches(img, kpts, scales, scale_th=min_scale_th, show_few=show)
    patches_r = get_patches(img_r, kpts_r, scales_r, scale_th=min_scale_th*real_scale, show_few=show)

    if compare:
        compare_patches(patches, patches_r, diffs)

    file_name_prefix = file_path[file_path.rfind("/") + 1:file_path.rfind(".")]
    for i in range(len(patches)):
        patch = patches_r[i]
        diff = (*diffs[i], patch.shape[0])
        file_name = "{}_{}.png".format(file_name_prefix, i)
        out_dict[file_name] = diff
        out_path = "{}/{}".format(out_dir, file_name)
        cv.imwrite(out_path, patch.numpy())


def prepare_data():

    repr_err_th = 2.0
    down_scale = 0.3

    max_items = None

    in_dir = "./dataset/raw_data"
    out_dir = "./dataset"

    if max_items:
        all = max_items
    else:
        all = len([fn for fn in os.listdir(in_dir) if fn.endswith(".tonemap.jpg")])

    data_dict = {}
    counter = 0
    for file_name in os.listdir(in_dir):

        if not file_name.endswith(".tonemap.jpg"):
            continue
        counter += 1
        if max_items and counter > max_items:
            break

        path = "{}/{}".format(in_dir, file_name)
        print("{}/{}".format(counter, all))
        process_patches_for_file(file_path=path,
                                 out_dir=out_dir,
                                 out_dict=data_dict,
                                 scale=down_scale,
                                 err_th=repr_err_th,
                                 compare=counter == 1,
                                 show=counter == 1)

    with open("{}/a_values.txt".format(out_dir), "w") as f:
        for k in data_dict:
            data = data_dict[k]
            f.write("{}, {}, {}, {}\n".format(k, data[0].numpy(), data[1].numpy(), data[2]))


# continue: encapsulate the params -> in some configurable object (torch-lightning)
if __name__ == "__main__":
    prepare_data()
