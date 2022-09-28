import copy
import glob
import numpy as np
import matplotlib.pyplot as plt
import wandb
from PIL import Image
import PIL
import torch
import os
import math
from config import *
from patch_dataset import get_error_stats, mean_abs_mean, DataRecord


def wand_log_me(msg, conf):
    if conf['enable_wandlog']:
        wandb.log(msg)


def mnn_generic(pts1, pts2, err_th):

    assert len(pts1.shape) == len(pts2.shape)

    pts1_r, pts2_r = pts1, pts2
    if len(pts1.shape) == 1:
        err_th *= math.sqrt(2)
        pts1_r, pts2_r = pts1[:, None].repeat(1, 2), pts2[:, None].repeat(1, 2)

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
    if mask1.shape[0] != mask2.shape[0]:
        print()

    return pts1_new, pts2_new, mask1, mask2


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

        fall_back = False
        if real_scale == 0.0 or math.fabs(real_scale - scale) > 0.1:
            if fall_back:
                wand_log_me("WARNING: scale={} => {}".format(real_scale, scale), config)
                real_scale = scale
            else:
                raise Exception("scale {} cannot be effectively realized for w, h = {}, {} in integer domain".format(scale, w, h))

    w_sc = round(w * real_scale)
    h_sc = round(h * real_scale)
    img_r = img.resize((h_sc, w_sc), resample=PIL.Image.LANCZOS)
    if show:
        show_pil(img_r)

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


def check_grey(img_np, detector, kpt_f):
    npa_grey = cv.cvtColor(img_np, cv.COLOR_BGR2GRAY);
    kpts_grey = detector.detect(npa_grey, mask=None)
    kpt_f_g = np.array([[kp.pt[1], kp.pt[0]] for kp in kpts_grey])
    check_COLOR_BGR2GRAY_all = kpt_f_g == kpt_f
    check_COLOR_BGR2GRAY = np.alltrue(check_COLOR_BGR2GRAY_all)
    npa_grey = cv.cvtColor(img_np, cv.COLOR_RGB2GRAY);
    kpts_grey = detector.detect(npa_grey, mask=None)
    kpt_f_g = np.array([[kp.pt[1], kp.pt[0]] for kp in kpts_grey])
    check_COLOR_RGB2GRAY_all = kpt_f_g == kpt_f
    check_COLOR_RGB2GRAY = np.alltrue(check_COLOR_RGB2GRAY_all)


def detect_kpts(img_np, scale_th, const_patch_size, config, check_const_patch_size=True):

    detector = get_detector(config)

    if check_const_patch_size and const_patch_size is not None:
             assert const_patch_size % 2 == 1, "doesn't work that way"

    h, w = img_np.shape[:2]

    kpts = detector.detect(img_np, mask=None)
    if len(kpts) == 0:
        return [], []

    kpt_f = np.array([[kp.pt[1], kp.pt[0]] for kp in kpts])
    kpt_i = np.round(kpt_f).astype(int)

    # this is to check against grey scale handling in SuperPoint
    check_grey(img_np, detector, kpt_f)

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

    # in [0, 360]
    orientations = torch.tensor([kpt.angle for kpt in kpts])[mask]
    return torch.from_numpy(kpt_f), torch.from_numpy(scales), orientations


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
    #log_me("original size: {}".format(img.size))
    return img


def possibly_refl_image(config, img):
    refl = config.get('reflection', None)
    if refl:
        refl = refl.upper()
    if refl == "XY":
        img = img.T
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
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def pil_img_transforms(img, config):
        img = np.array(img)
        img = possibly_to_grey_scale(config, img)
        img = possibly_refl_image(config, img)
        return img


def get_img_tuple(path, scale, config, show=False):

    img = get_pil_img(path, show)
    img_r, real_scale = scale_pil(img, scale=scale, config=config, show=show)
    img = pil_img_transforms(img, config)
    img_r = pil_img_transforms(img_r, config)

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


def augment_and_write_patch(patch, dr, file_name_prefix, out_map, out_dir, write_data=True):

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
        img_out_path = "{}/data/{}".format(out_dir, file_name)
        if write_data:
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

    # convert and show the image
    img_orig_pil = get_pil_img(file_path)
    img_orig = pil_img_transforms(img_orig_pil)

    kpts, kpt_scales, _ = detect_kpts(img_orig, min_scale_th, const_patch_size=None, config=config)
    if len(kpts) == 0:
        return

    counter = 0
    matched = set()
    for kpt_scale_index, kpt_scale in enumerate(kpt_scales):

        if max_items and counter > max_items:
            break

        kpts_orig = kpts
        kpt_scales_orig = kpt_scales

        # FIXME test *2?
        scale = (const_patch_size / scale_ratio_th) / kpt_scale.item()

        img_r, real_scale = scale_pil(img_orig_pil, scale, config=config)
        img_r = pil_img_transforms(img_r, config)

        kpts_r, kpt_scales_r, _ = detect_kpts(img_r, min_scale_th*real_scale, const_patch_size, config)
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
                mask[i] = True
        kpts_orig, kpt_scales_orig, kpts_r, kpt_scales_r, diffs, scale_ratios = apply_mask(mask, [kpts_orig, kpt_scales_orig, kpts_r, kpt_scales_r, diffs, scale_ratios])
        patches = get_patches(img_orig, kpts_orig, kpt_scales_orig, const_patch_size, config)
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
                img_scale_y=img_r.shape[0]/img_orig.shape[0],
                img_scale_x=img_r.shape[0]/img_orig.shape[0],
                original_img_size_y=img_orig.shape[0],
                original_img_size_x=img_orig.shape[1],
                resized_img_size_y=img_r.shape[0],
                resized_img_size_x=img_r.shape[1],
                augmented=None
            )
            write_data = config['write_data']
            augment_and_write_patch(patch,
                                    dr,
                                    "{}_{}".format(file_name_prefix, i),
                                    out_map,
                                    out_dir,
                                    write_data)


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

    # convert and show the image
    img, img_r, real_scale = get_img_tuple(file_path, scale, config)

    kpts, kpt_scales, _ = detect_kpts(img, min_scale_th, const_patch_size, config)
    kpts_r, kpt_scales_r, _ = detect_kpts(img_r, min_scale_th*real_scale, const_patch_size, config)

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
        write_data = config['write_data']
        augment_and_write_patch(patch,
                                dr,
                                "{}_{}".format(file_name_prefix, i),
                                out_map,
                                out_dir,
                                write_data)


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


def prepare_and_clean_dir(dataset_config):
    out_dir = get_full_ds_dir(dataset_config)
    clean = dataset_config['clean_out_dir']
    write_data = dataset_config['write_data']
    if write_data:
        data_dir_path = "{}/data".format(out_dir)
        if not os.path.exists(data_dir_path):
            os.makedirs(data_dir_path, exist_ok=True)
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


def possibly_write_data(dataset_config, out_map):
    write_data = dataset_config['write_data']
    if write_data:
        out_dir = get_full_ds_dir(dataset_config)
        with open("{}/a_metadata.txt".format(out_dir), "w") as md_file:
            write_metada(md_file, out_map, dataset_config)

        with open("{}/a_values.txt".format(out_dir), "w") as md_file:
            write_metada(md_file, out_map, dataset_config)

            md_file.write("# schema: {}\n".format(DataRecord.schema()))
            for (fn, value) in out_map.items():
                to_write = "{}, {}\n".format(fn, value.line_str())
                md_file.write(to_write)


def read_img(file_path, config):
    img_pil = get_pil_img(file_path)
    img_np = pil_img_transforms(img_pil, config)
    return img_np, img_pil


def reflection_experiment(file_path, config):

    img_np_o, img_pil = read_img(file_path, config)
    kpts_0, _, _ = detect_kpts(img_np_o, 0.0, 0.0, config, False)

    img_xy = img_np_o.T
    img_y = np.flip(img_np_o, 0).copy()

    # given img_x
    img_x = np.flip(img_np_o, 1).copy()
    kpts_1, _, _ = detect_kpts(img_x, 0.0, 0.0, config, False)

    print("number of orig kpts: {}, {}".format(kpts_0.shape[0], kpts_1.shape[0]))

    kpts_0_new_1d, kpts_1_new_1d, mask_0, mask_1 = mnn_generic(kpts_0[:, 0], kpts_1[:, 0], err_th=1.0)

    print("number of new kpts: {}, {}".format(kpts_0_new_1d.shape[0], kpts_1_new_1d.shape[0]))
    distances = (kpts_1_new_1d - kpts_0_new_1d).abs() # only one dimensional
    print("distances in 1 D - min: {}, max: {}".format(distances.min(), distances.max()))

    # verify ?
    kpts_0_new, kpts_1_new = kpts_0[mask_0], kpts_1[mask_1]
    centers_flip_axis = (kpts_0_new[:, 1] + kpts_1_new[:, 1]) / 2
    print("symmetry on the other axis (shape={}): min: {}, max: {}".format(img_np_o.shape[1], centers_flip_axis.min(), centers_flip_axis.max()))
    # histogram?
    mean_centers = centers_flip_axis.mean()
    var_centers = centers_flip_axis.var()
    print("centers mean: {}, var: {}".format(mean_centers, var_centers))


def reflection_experiment_loop():

    dataset_config = get_config()['dataset']
    in_dirs = dataset_config['in_dirs']
    keys = dataset_config['keys']

    ends_with = dataset_config['ends_with']
    max_items = dataset_config['max_items']

    all_file_counter = 0
    out_map = {}
    for i, in_dir in enumerate(in_dirs):

        cur_dir_files = [fn for fn in os.listdir(in_dir) if fn.endswith(ends_with)]
        new_len = min(max_items - all_file_counter, len(cur_dir_files))
        cur_dir_files = cur_dir_files[:new_len]
        all_file_counter += new_len

        print("Processing dir {}: {}/{}".format(in_dir, i, len(in_dirs)))
        key = keys[i]

        for file_name in cur_dir_files:
            path = "{}/{}".format(in_dir, file_name)
            reflection_experiment(path, dataset_config)


def prepare_data(dataset_config, in_dirs, keys):

    ends_with = dataset_config['ends_with']
    max_items = dataset_config['max_items']
    const_patch_size = dataset_config.get('const_patch_size')
    if const_patch_size is not None:
            assert const_patch_size % 2 == 1, "doesn't work that way"

    prepare_and_clean_dir(dataset_config)

    out_map = {}
    all = "?1?" if max_items is not None else None

    for i, in_dir in enumerate(in_dirs):

        counter = 0

        if not max_items:
            all = len([fn for fn in os.listdir(in_dir) if fn.endswith(ends_with)])

        print("Processing dir {}: {}/{}".format(in_dir, i, len(in_dirs)))
        key = keys[i]

        for file_name in os.listdir(in_dir):

            if not file_name.endswith(ends_with):
                continue
            counter += 1
            if max_items and counter > max_items:
                break

            path = "{}/{}".format(in_dir, file_name)
            print("Processing file {}: {}/{}, learning examples: {}".format(path, counter, all, len(out_map)))
            process_patches_for_file(file_path=path,
                                     config=dataset_config,
                                     out_map=out_map,
                                     key=key,
                                     max_items=max_items)

    possibly_write_data(dataset_config, out_map)
    return list(out_map.items())


def write_metada(file, out_map, config):

    detector_name = config['detector'].upper()

    def print_min_max_stat(file, stat, name):
        file.write("# {} {} (min, max): ({}, {})\n".format(detector_name, name, stat[0], stat[1]))

    def print_m_am_stat(file, stat, name, leave_abs_mean=False, leave_std_dev=False):
        mean, abs_mean = mean_abs_mean(stat)
        std_dev = np.sqrt(stat.var(axis=0))

        file.write("# {} mean {} error: {}\n".format(detector_name, name, mean))
        if not leave_abs_mean:
            file.write("# {} absolute mean {} error: {}\n".format(detector_name, name, abs_mean))
        if not leave_std_dev:
            file.write("# {} std dev of {} error: {}\n".format(detector_name, name, std_dev))

    file.write("# entries: {}\n".format(len(out_map)))

    distances, errors, angles = get_error_stats(out_map.items())
    print_m_am_stat(file, distances, "distance", leave_abs_mean=True)
    print_m_am_stat(file, distances ** 2, "distance squared", leave_abs_mean=True)
    print_m_am_stat(file, errors, "")
    print_m_am_stat(file, angles, "angle")

    patch_size_min_max, scale_min_max, scale_ratio_min_max = get_ds_stats(out_map.items())
    print_min_max_stat(file, patch_size_min_max, "patch size")
    print_min_max_stat(file, scale_min_max, "original scale")
    print_min_max_stat(file, scale_ratio_min_max, "scale ratio")

    file.write("### CONFIG ###\n")
    for k, v in list(config.items()):
        file.write("#\t\t\t{}: {}\n".format(k, v))
    file.write("### CONFIG ###\n")


def get_wand_name(config, entry_list):

    wandb_tags_keys = config['wandb_tags_keys']
    name = ""
    for wandb_tags_key in wandb_tags_keys:
        if wandb_tags_key == "magic_items":
            name = name + ":items=" + str(len(entry_list))
        elif wandb_tags_key.startswith("no_key"):
            wandb_tags_key = wandb_tags_key[7:]
            value = config.get(wandb_tags_key, None)
            if value:
                name = name + ":" + str(value)
        else:
            value = config.get(wandb_tags_key, None)
            if value:
                name = name + ":{}={}".format(wandb_tags_key, str(value))

    return name


def prepare_data_by_scale(wand_project="mean_std_dev"):

    config = get_config()['dataset']
    in_dirs = config['in_dirs']
    keys = config['keys']

    values = [scale_int / 10 for scale_int in range(1, 10)]
    means = []
    std_devs = []

    for scale in values:
        config['down_scale'] = scale
        dn = config['detector']
        config['out_dir'] = "dataset/{}_int_scale_{}_size_".format(dn, scale).replace(".", "_")
        print(list(config.items()))
        entry_list = prepare_data(config, in_dirs, keys)
        _, errors, _ = get_error_stats(entry_list)
        mean, _ = mean_abs_mean(errors)
        means.append(mean.tolist())
        std_dev = np.sqrt(errors.var(axis=0)).tolist()
        std_devs.append(std_dev)

    std_devs = np.array(std_devs)
    means = np.array(means)

    if wand_project:

        # tags...
        wandb.init(project=wand_project, name=get_wand_name(config, entry_list))

        data = [[x, y] for (x, y) in zip(values, means[:, 1])]
        table = wandb.Table(data=data, columns=["scale", "mean error x"])
        wandb.log({"mean error x": wandb.plot.line(table, "scale", "mean error x", title="Mean error(x) as a function of scale")})

        data = [[x, y] for (x, y) in zip(values, means[:, 0])]
        table = wandb.Table(data=data, columns=["scale", "mean error y"])
        wandb.log({"mean error y": wandb.plot.line(table, "scale", "mean error y", title="Mean error(y) as a function of scale")})

        data = [[x, y] for (x, y) in zip(values, std_devs[:, 1])]
        table = wandb.Table(data=data, columns=["scale", "std dev x"])
        wandb.log({"std dev(x)": wandb.plot.line(table, "scale", "std dev x", title="Std dev(x) of error as a function of scale")})

        data = [[x, y] for (x, y) in zip(values, std_devs[:, 0])]
        table = wandb.Table(data=data, columns=["scale", "std dev y"])
        wandb.log({"std dev(y)": wandb.plot.line(table, "scale", "std dev y", title="Std dev(y) of error as a function of scale")})


def simple_prepare_data():
    config = get_config()['dataset']
    in_dirs = config['in_dirs']
    keys = config['keys']
    prepare_data(config, in_dirs, keys)


if __name__ == "__main__":
    # prepare_data_by_scale()
    # simple_prepare_data()
    reflection_experiment_loop()
