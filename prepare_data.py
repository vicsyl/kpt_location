import copy
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
from patch_dataset import get_error_stats, mean_abs_mean, DataRecord


# FIXME: use proper logging
def log_me(s):
    print(s)


def mnn(kpts, kpts_scales, kpts_r, kpts_r_scales, scale, config):

    err_th = config['err_th']
    scale_ratio_th = config['scale_ratio_th']
    half_pixel_adjusted = config['half_pixel_adjusted']
    #kpts_reprojected_or = kpts_r / scale
    if half_pixel_adjusted:
        magic_scale = scale / 2 # hypothesis
        adjustment = torch.ones(2) * magic_scale
        kpts_r = kpts_r + adjustment

    kpts_reprojected = kpts_r / scale
    #kpts_reprojected_adj = kpts_reprojected - kpts_reprojected_or

    d_mat = torch.cdist(kpts, kpts_reprojected)

    min0 = torch.min(d_mat, dim=0)
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
        assert torch.allclose(ds, dists)

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

    return kpts0, kpts_scales, kpts1, kpts_r_scales, diffs, scale_ratios


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


def detect_kpts(img_np, scale_th, const_patch_size, config):

    detector = get_detector(config)

    if const_patch_size is not None:
            assert const_patch_size % 2 == 1, "doesn't work that way"

    # npa_grey = cv.cvtColor(npa, cv.COLOR_BGR2GRAY);
    # kpts_grey = detector.detect(npa_grey, mask=None)
    # kpt_f_g = np.array([[kp.pt[1], kp.pt[0]] for kp in kpts_grey])
    # check = np.alltrue(kpt_f_g == kpt_f)

    h, w = img_np.shape[:2]

    kpts = detector.detect(img_np, mask=None)
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

    show = config['show_kpt_patches']
    if show:
        npac = img_np.copy()
        cv.drawKeypoints(img_np, kpts, npac, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure()
        plt.title("kpts")
        plt.imshow(npac)
        plt.show()
        plt.close()

    return torch.from_numpy(kpt_f), torch.from_numpy(scales)


def print_and_check_margins(kpt_i, margins_np, img_t):
    print()
    for i, kp_i in list(enumerate(kpt_i)):
        t = ((kp_i[0] - margins_np[i]).item(),
              (kp_i[0] + margins_np[i]).item() + 1,
              (kp_i[1] - margins_np[i]).item(),
              (kp_i[1] + margins_np[i]).item() + 1)
        print(t, img_t.shape)


def show_patches(patches, label, config, detect):

    cols = 5
    rows = 5

    if detect:
        detector = get_detector(config)

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


def get_patches(img_np, kpt_f, kpt_scales, const_patch_size, config):
    """
    :param img_np:
    :param kpt_f:
    :param kpt_scales:
    :param scale_th:
    :param show_few:
    :return: patches: List[Tensor]
    """

    img_t = torch.tensor(img_np)

    kpt_i = torch.round(kpt_f).to(torch.int)
    if const_patch_size is not None:
        assert const_patch_size % 2 == 1, "doesn't work that way"
        margins_np = np.ones(kpt_scales.shape[0], dtype=int) * const_patch_size // 2
    else:
        margins = torch.ceil(kpt_scales / 2.0)
        margins_np = margins.to(torch.int).numpy()
    # print_and_check_margins(kpt_i, margins_np, img_t)

    patches = slice_patches(img_t, kpt_i, margins_np)

    show = config['show_kpt_patches']
    if show:
        detector = get_detector(config)
        kpts = detector.detect(img_np, mask=None)
        npac = img_np.copy()
        cv.drawKeypoints(img_np, kpts, npac, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_tc = torch.tensor(npac)
        ps_kpts = slice_patches(img_tc, kpt_i, margins_np)

        show_patches(ps_kpts, "Patches - original", config, detect=False)
        show_patches(patches, "Patches - redetected", config, detect=True)

    return patches


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


def get_pil_img(path, show=False):

    img = Image.open(path)
    if show:
        show_pil(img)
    log_me("original size: {}".format(img.size))
    return img


def get_img_tuple(path, scale, config, show=False):

    img = get_pil_img(path, show)

    img_r, real_scale = scale_pil(img, scale=scale, config=config, show=show)
    img = np.array(img)
    img_r = np.array(img_r)

    to_grey_scale = config['to_grey_scale']
    if to_grey_scale:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)
    return img, img_r, real_scale


def apply_mask(mask, np_list):
    return tuple([npa[mask] for npa in np_list])


def augment_patch(patch, diffs):
    patch_r_y = torch.flip(patch, dims=[0])
    diffs_r_y = -diffs[0], diffs[1]
    patch_r_x = torch.flip(patch, dims=[1])
    diffs_r_x = diffs[0], -diffs[1]

    patches = [patch]
    diff_l = [diffs]
    augment_keys = ["original"]
    for i in range(3):
        patch = torch.rot90(patch, 1, [0, 1])
        patches.append(patch)
        diffs = -diffs[1], diffs[0]
        diff_l.append(diffs)
        augment_keys.append("rotated_{}".format((i + 1) * 90))
    patches = patches + [patch_r_y, patch_r_x]
    diff_l = diff_l + [diffs_r_y, diffs_r_x]
    augment_keys = augment_keys + ["reflected_y", "reflected_x"]
    return patches, diff_l, augment_keys


def augment_and_write_patch(patch, dr, file_name_prefix, out_map, out_dir):

    patches_aug, diffs_aug, augmented_keys = augment_patch(patch, (dr.dy, dr.dx))
    for index, patch_aug in enumerate(patches_aug):
        dy, dx = diffs_aug[index]
        if index > 0:
            dr = copy.copy(dr)
        else:
            assert dr.dy == dy and dr.dx == dx
        dr.dy, dr.dx = dy, dx
        dr.augmented = augmented_keys[index]
        file_name = "{}_{}.png".format(file_name_prefix, index)
        out_map[file_name] = dr
        img_out_path = "{}/{}".format(out_dir, file_name)
        cv.imwrite(img_out_path, patch_aug.numpy())


def process_patches_for_file_dynamic(file_path,
                                     config,
                                     out_map,
                                     key="",
                                     max_items=None):

    out_dir = get_full_ds_dir(config)
    min_scale_th = config['min_scale_th']
    const_patch_size = config.get('const_patch_size')
    compare = config['compare_patches']
    scale_ratio_th = config['scale_ratio_th']

    print("Processing: {}".format(file_path))

    # convert and show the image
    pil_img_orig = get_pil_img(file_path)
    np_img_orig = np.array(pil_img_orig)
    to_grey_scale = config['to_grey_scale']
    if to_grey_scale:
        np_img_orig = cv.cvtColor(np_img_orig, cv.COLOR_BGR2GRAY)

    kpts, kpt_scales = detect_kpts(np_img_orig, min_scale_th, const_patch_size=None, config=config)
    if len(kpts) == 0:
        return

    counter = 0
    matched = set()
    for kpt_scale_index, kpt_scale in enumerate(kpt_scales):

        if max_items and counter > max_items:
            break

        kpts_orig = kpts
        kpt_scales_orig = kpt_scales

        cur_kpt = kpts_orig[kpt_scale_index]
        print("scale for kpt: {}".format(cur_kpt))

        # FIXME test *2?
        scale = (const_patch_size / scale_ratio_th) / kpt_scale.item()

        img_r, real_scale = scale_pil(pil_img_orig, scale, config=config)
        # FIXME dry
        img_r = np.array(img_r)
        if to_grey_scale:
            img_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

        kpts_r, kpt_scales_r = detect_kpts(img_r, min_scale_th*real_scale, const_patch_size, config)
        if len(kpts_r) == 0:
            continue

        kpts_orig, kpt_scales_orig, kpts_r, kpt_scales_r, diffs, scale_ratios = mnn(kpts_orig, kpt_scales_orig, kpts_r, kpt_scales_r, real_scale, config)

        #check kpt_scales_r
        mask = (kpt_scales_r < const_patch_size) & (kpt_scales_r > const_patch_size - 2)
        kpts_orig, kpt_scales_orig, kpts_r, kpt_scales_r, diffs, scale_ratios = apply_mask(mask, [kpts_orig, kpt_scales_orig, kpts_r, kpt_scales_r, diffs, scale_ratios])
        mask = np.zeros(kpts_orig.shape[0], dtype=bool)
        for i, kpt in enumerate(kpts_orig):
            if not matched.__contains__(kpt):
                matched.add(kpt)
                print("matched kpt: {}".format(kpt))
                mask[i] = True
        kpts_orig, kpt_scales_orig, kpts_r, kpt_scales_r, diffs, scale_ratios = apply_mask(mask, [kpts_orig, kpt_scales_orig, kpts_r, kpt_scales_r, diffs, scale_ratios])
        patches = get_patches(np_img_orig, kpts_orig, kpt_scales_orig, const_patch_size, config)
        patches_r = get_patches(img_r, kpts_r, kpt_scales_r, const_patch_size, config)
        if compare:
            compare_patches(patches, patches_r, diffs)

        if key != "":
            key = key + "_"
        file_name_prefix = key + file_path[file_path.rfind("/") + 1:file_path.rfind(".")] + "_" + str(kpt_scale_index)
        for i in range(len(patches)):
            counter += 1
            if max_items and counter > max_items:
                break
            patch = patches_r[i]
            dr = DataRecord(
                dy=diffs[i][0].item(),
                dx=diffs[i][1].item(),
                patch_size=patch.shape[0],
                kpt_orig_scale=kpt_scales_orig[i].item(),
                kpt_resize_scale=kpt_scales_r[i].item(),
                scale_ratio=scale_ratios[i].item(),
                real_scale=real_scale,
                img_scale_y=img_r.shape[0]/np_img_orig.shape[0],
                img_scale_x=img_r.shape[0]/np_img_orig.shape[0],
                original_img_size_y=np_img_orig.shape[0],
                original_img_size_x=np_img_orig.shape[1],
                resized_img_size_y=img_r.shape[0],
                resized_img_size_x=img_r.shape[1],
                augmented=None
            )
            augment_and_write_patch(patch,
                                    dr,
                                    "{}_{}".format(file_name_prefix, i),
                                    out_map,
                                    out_dir)


def process_patches_for_file(file_path,
                             config,
                             out_map,
                             key="",
                             max_items=None):

    dynamic_resizing = config['dynamic_resizing']
    if dynamic_resizing:
        process_patches_for_file_dynamic(file_path, config, out_map, key, max_items)
    else:
        process_patches_for_file_simple(file_path, config, out_map, key, max_items)


def process_patches_for_file_simple(file_path,
                                    config,
                                    out_map,
                                    key="",
                                    max_items=None):

    scale = config['down_scale']
    out_dir = get_full_ds_dir(config)
    min_scale_th = config['min_scale_th']
    const_patch_size = config.get('const_patch_size')
    compare = config['compare_patches']

    print("Processing: {}".format(file_path))

    # convert and show the image
    img, img_r, real_scale = get_img_tuple(file_path, scale, config)

    kpts, kpt_scales = detect_kpts(img, min_scale_th, const_patch_size, config)
    kpts_r, kpt_scales_r = detect_kpts(img_r, min_scale_th*real_scale, const_patch_size, config)

    if len(kpts) == 0 or len(kpts_r) == 0:
        return

    kpts, kpt_scales, kpts_r, kpt_scales_r, diffs, scale_ratios = mnn(kpts, kpt_scales, kpts_r, kpt_scales_r, real_scale, config)

    patches = get_patches(img, kpts, kpt_scales, const_patch_size, config)
    patches_r = get_patches(img_r, kpts_r, kpt_scales_r, const_patch_size, config)

    if compare:
        compare_patches(patches, patches_r, diffs)

    if key != "":
        key = key + "_"
    file_name_prefix = key + file_path[file_path.rfind("/") + 1:file_path.rfind(".")]
    for i in range(len(patches)):
        if max_items and i > max_items:
            break
        patch = patches_r[i]
        dr = DataRecord(
            dy=diffs[i][0].item(),
            dx=diffs[i][1].item(),
            patch_size=patch.shape[0],
            kpt_orig_scale=kpt_scales[i].item(),
            kpt_resize_scale=kpt_scales_r[i].item(),
            scale_ratio=scale_ratios[i].item(),
            real_scale=real_scale,
            img_scale_y=img_r.shape[0] / img.shape[0],
            img_scale_x=img_r.shape[0] / img.shape[0],
            original_img_size_y=img.shape[0],
            original_img_size_x=img.shape[1],
            resized_img_size_y=img_r.shape[0],
            resized_img_size_x=img_r.shape[1],
            augmented=None
        )
        augment_and_write_patch(patch,
                                dr,
                                "{}_{}".format(file_name_prefix, i),
                                out_map,
                                out_dir)


def get_ds_stats(entries):
    def adjust_min_max(min_max_stat, value):
        if min_max_stat[0] is None or min_max_stat[0] > value:
            min_max_stat[0] = value
        if min_max_stat[1] is None or min_max_stat[1] < value:
            min_max_stat[1] = value
        return min_max_stat

    patch_size_min_max = [None, None]
    scale_min_max = [None, None]
    scale_ratio_min_max = [None, None]
    for _, data_record in entries:
        patch_size_min_max = adjust_min_max(patch_size_min_max, data_record.patch_size)
        scale_min_max = adjust_min_max(scale_min_max, data_record.kpt_orig_scale)
        scale_ratio_min_max = adjust_min_max(scale_ratio_min_max, data_record.scale_ratio)

    return patch_size_min_max, scale_min_max, scale_ratio_min_max


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
    all = "?1?" if max_items is not None else None

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
                                     max_items=max_items)

    def print_min_max_stat(file, stat, name):
        file.write("# detector {} (min, max): ({}, {})\n".format(name, stat[0], stat[1]))

    def print_m_am_stat(file, stat, name, leave_abs_mean=False):
        mean, abs_mean = mean_abs_mean(stat)
        if leave_abs_mean:
            file.write("# detector mean {} error: {}\n".format(name, mean))
        else:
            file.write("# detector mean and absolute mean {} error: {}, {}\n".format(name, mean, abs_mean))

    with open("{}/a_values.txt".format(out_dir), "w") as md_file:
        md_file.write("# entries: {}\n".format(len(out_map)))

        distances, errors, angles = get_error_stats(out_map.items())
        print_m_am_stat(md_file, distances, "distance", leave_abs_mean=True)
        print_m_am_stat(md_file, errors, "")
        print_m_am_stat(md_file, angles, "angle")

        patch_size_min_max, scale_min_max, scale_ratio_min_max = get_ds_stats(out_map.items())
        print_min_max_stat(md_file, patch_size_min_max, "patch size")
        print_min_max_stat(md_file, scale_min_max, "original scale")
        print_min_max_stat(md_file, scale_ratio_min_max, "scale ratio")

        md_file.write("# schema: {}\n".format(DataRecord.schema()))
        for (fn, value) in out_map.items():
            to_write = "{}, {}\n".format(fn, value.line_str())
            md_file.write(to_write)


if __name__ == "__main__":

    config = get_config()['dataset']
    in_dirs = config['in_dirs']
    keys = config['keys']

    prepare_data(config, in_dirs, keys)
