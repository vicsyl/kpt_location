import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import PIL
import torch


def log_me(s):
    print(s)


def mnn(kpts, kpts_r, scale, th):

    kpts_reproj = kpts_r / scale
    d_mat = torch.cdist(kpts, kpts_reproj)
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
    kpts1 = kpts_r[mask_th]
    dists = min0[0][mask_th]
    diffs = kpts_reproj[mask_th] - kpts0
    if verify:
        ds = torch.diag(torch.cdist(kpts0, kpts_reproj[mask_th]))
        assert torch.all(ds == dists)

    return kpts0, kpts1, diffs


def scale_pil(img, scale, show=False):
    """
    :param img:
    :param scale: in [0, 1]
    :return:
    """
    h, w = img.size
    gcd = gcd_euclid(w, h)

    real_scale_gcd = round(gcd * scale)
    # log_me("gcd: {}".format(gcd))
    # log_me("real_scale_gcd: {}".format(real_scale_gcd))

    w_sc = int(w / gcd * real_scale_gcd)
    h_sc = int(h / gcd * real_scale_gcd)
    img_r = img.resize((h_sc, w_sc), resample=PIL.Image.LANCZOS)
    log_me("scaled to: {}".format(img_r.size))
    if show:
        show_pil(img_r)

    real_scale = real_scale_gcd / gcd
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


def detect(img_pil, patch_size, detector=get_default_detector(), show=False):

    npa = np.array(img_pil)
    h, w, c = npa.shape

    kpts = detector.detect(npa, mask=None)

    kpt_f = np.array([kp.pt for kp in kpts])

    kpt_i = np.round(kpt_f).astype(np.int)
    margin = patch_size // 2
    mask = (kpt_i[:, 0] >= margin) & (kpt_i[:, 1] >= margin)
    mask = mask & (kpt_i[:, 0] < h - margin) & (kpt_i[:, 1] < w - margin)
    kpt_f = kpt_f[mask]

    if show:
        npac = npa.copy()
        cv.drawKeypoints(npa, kpts, npac, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure()
        plt.imshow(npac)
        plt.show()

    return torch.from_numpy(kpt_f)


def get_patches(img_pil, patch_size, kpt_f, exclude_near_edge=True, show_few=False):

    assert patch_size % 2 == 1, "uncentered patches"
    if not exclude_near_edge:
        raise "Not implemented"

    #npa = torch.tensor(img_pil)
    img_t = torch.tensor(np.array(img_pil))
    kpt_i = torch.round(kpt_f).to(torch.int)
    margin = patch_size // 2

    patches_l = [img_t[kp_i[0] - margin: kp_i[0] + margin, kp_i[1] - margin: kp_i[1] + margin][None] for kp_i in kpt_i]
    patches = torch.cat(patches_l, dim=0)
    # patches = np.array([img_t[kp_i[0] - margin: kp_i[0] + margin,
    #                         kp_i[1] - margin: kp_i[1] + margin] for kp_i in kpt_i])

    if show_few:
        cols = 4
        rows = 4
        fig, axs = plt.subplots(rows, cols, figsize=(5, 5))
        fig.suptitle("Few patches")

        for ix in range(rows):
            for iy in range(cols):
                #axs[ix, iy].set_title("foo")
                axs[ix, iy].set_axis_off()
                axs[ix, iy].imshow(patches[ix * cols + iy].numpy())
        plt.show()

    return patches


def compare_patches(patches0, patches1, diffs):

    cols = 8
    cols = min(cols, patches0.shape[0])

    fig, axs = plt.subplots(2, cols, figsize=(12, 4))
    fig.suptitle("Patches comparison")

    for n in range(cols):
        for r in range(2):
            if r == 0:
                axs[r, n].set_title("diff=\n{:.02f},{:.02f}".format(diffs[n, 0].item(), diffs[n, 1].item()))
            axs[r, n].set_axis_off()
            patch = patches0[n] if r == 0 else patches1[n]
            axs[r, n].imshow(patch.numpy())
    plt.show()

