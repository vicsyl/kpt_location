from patches import *


# def convert_pil(img_plt):
#     """out of curiosity"""
#     npar = np.array(img_plt)
#     plt.imshow(npar)
#
#     timg1 = KR.image_to_tensor(npar, False).float() / 255.
#     # timg1 = KR.color.bgr_to_grayscale(timg1)
#
#     img_b = KR.tensor_to_image(timg1)
#     plt.figure()
#     plt.imshow(img_b)


def get_img_tuple(path, scale):

    img = Image.open(path)
    show_pil(img)
    log_me("PIL size: {}".format(img.size))

    img_r, real_scale = scale_pil(img, scale=scale, show=True)
    return img, img_r, real_scale


def main():
    patch_size = 33

    # convert and show the image
    img, img_r, real_scale = get_img_tuple("work/frame.0000.tonemap.jpg", scale=0.3)

    kpts = detect(img, patch_size=patch_size, show=True)
    kpts_r = detect(img_r, patch_size=patch_size, show=True)

    kpts, kpts_r, diffs = mnn(kpts, kpts_r, real_scale, th=2)

    patches = get_patches(img, patch_size=33, kpt_f=kpts, show_few=True)
    patches_r = get_patches(img_r, patch_size=33, kpt_f=kpts_r, show_few=True)

    compare_patches(patches, patches_r, diffs)


if __name__ == "__main__":
    main()


# comments
# class kornia.feature.SIFTFeature(num_features=8000, upright=False, rootsift=True, device=torch.device('cpu'))
