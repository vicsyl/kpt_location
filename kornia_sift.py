import numpy as np
import torch

from kornia.feature import ScaleSpaceDetector, BlobDoG, LAFOrienter, PassLAF, LAFDescriptor, SIFTDescriptor
from kornia.feature.laf import scale_laf
from kornia.geometry.subpix import ConvQuadInterp3d
from kornia.geometry.transform import ScalePyramid
import kornia.utils as KU

import cv2 as cv

from sift_detectors import BaseDescriptor
from scale_pyramid import MyScalePyramid

default_nearest_scale_pyramid = ScalePyramid(3, 1.6, 32, double_image=True)
lin_interpolation_scale_pyramid = MyScalePyramid(3, 1.6, 32, double_image=True)


class NumpyKorniaSiftDescriptor(BaseDescriptor):

    def __str__(self):
        return f"SIFT kornia {self.interpolation_mode} {self.adjustment.cpu().numpy()}"

    """
    see kornia.feature.integrated.SIFTFeature
    plus num_features is different (originally 8000) and the ScalePyramid can be overriden for obvious reasons
    """
    def __init__(self, upright=False, num_features=500, interpolation_mode='nearest', rootsift=True, adjustment=[0.0, 0.0], scale_pyramid=None):
        super().__init__()
        self.interpolation_mode = interpolation_mode if not scale_pyramid else "via_scale_pyr"
        if not scale_pyramid:
            scale_pyramid = MyScalePyramid(3, 1.6, 32, double_image=True, interpolation_mode=interpolation_mode)
        #     scale_pyramid = default_nearest_scale_pyramid if nearest else lin_interpolation_scale_pyramid

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.adjustment = torch.tensor(adjustment, device=device)
        self.detector = ScaleSpaceDetector(
            num_features=num_features,
            resp_module=BlobDoG(),
            nms_module=ConvQuadInterp3d(10),
            scale_pyr_module=scale_pyramid,
            ori_module=PassLAF() if upright else LAFOrienter(19),
            scale_space_response=True,
            minima_are_also_good=True,
            mr_size=6.0,
        )
        patch_size = 41
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.descriptor = LAFDescriptor(
            SIFTDescriptor(patch_size=patch_size, rootsift=rootsift), patch_size=patch_size, grayscale_descriptor=True
        ).to(self.device)
        self.detector.eval()
        self.scaling_coef = 1.0

    def set_rotate_gauss(self, rotate90_gauss):
        self.detector.scale_pyr.rotate90_gauss = rotate90_gauss

    def set_rotate_interpolation(self, rotate90_interpolation):
        self.detector.scale_pyr.rotate90_interpolation = rotate90_interpolation

    def get_lafs_responses(self, img_np, mask=None):
        # FIXME handle greyscale consistently
        # NOTE a simple check on number of dims would suffice here for greyscale option being on,
        # but the visualization won't work
        if len(img_np.shape) == 2:
            img_np = img_np[:, :, None]
        else:
            img_np = cv.cvtColor(img_np, cv.COLOR_BGR2GRAY)
        with torch.no_grad():
            img_t3 = KU.image_to_tensor(img_np, False).float() / 255.
            img_t3 = img_t3.to(device=self.device)
            laffs, responses = self.detector(img_t3, mask)
            laffs[0, :, :, 2] = laffs[0, :, :, 2] + self.adjustment
            return laffs, responses, img_t3

    def cv_kpt_from_laffs_responses(self, laffs, responses):
        kpts = []
        for i, response in enumerate(responses[0]):
            yx = laffs[0, i, :, 2]
            kp = cv.KeyPoint(yx[0].item(), yx[1].item(), response.item(), angle=0)
            kpts.append(kp)
        return kpts

    def detect(self, img_np, mask=None):
        laffs, responses, _ = self.get_lafs_responses(img_np, mask)
        kpts = self.cv_kpt_from_laffs_responses(laffs, responses)
        return kpts

    def detect_compute_measure(self, img, mask):

        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            kpts_other, desc_other = self.detectAndCompute(img, mask)
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()

            time = start.elapsed_time(end) / 1000
            return kpts_other, desc_other, time
        else:
            return super().detect_compute_measure(img, mask)

    def detectAndCompute(self, img, mask):
        lafs, responses, img_t = self.get_lafs_responses(img, mask)
        kpts = self.cv_kpt_from_laffs_responses(lafs, responses)
        lafs = scale_laf(lafs, self.scaling_coef)
        with torch.no_grad():
            descs = self.descriptor(img_t, lafs)
            descs = descs[0].cpu().numpy()
        # TODO
        return kpts, descs
