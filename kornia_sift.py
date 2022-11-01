import math
import torch
from kornia.feature import ScaleSpaceDetector, BlobDoG, LAFOrienter, PassLAF
from kornia.geometry.subpix import ConvQuadInterp3d
from kornia.geometry.transform import ScalePyramid
import kornia.utils as KU

import cv2 as cv


class NumpyKorniaSiftDetector:
    def __init__(self, upright=False, num_features=500, scale_pyramid=ScalePyramid(3, 1.6, 32, double_image=True)):
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
        self.detector.eval()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def detect(self, img_np, mask=None):

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
            kpts = []
            for i, response in enumerate(responses[0]):
                yx = laffs[0, i, :, 2]
                kp = cv.KeyPoint(yx[0].item(), yx[1].item(), response.item(), angle=0)
                kpts.append(kp)
            return kpts
