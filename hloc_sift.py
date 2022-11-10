import cv2 as cv
import kornia.utils as KU
import numpy as np
# from hloc.extractors.dog import DoG


class HlocSiftDescriptor:

    def __str__(self) -> str:

        return f"HLOC SIFT: {self.abs_adjustment[0]}"

    def __init__(self, adjustment=[0.0, 0.0]):
        self.dog = None # use the line below
        # self.dog = DoG(DoG.default_conf)
        self.abs_adjustment = np.array(adjustment)

    def adjust_cv_kpts(self, cv_kpts):
        for cv_kpt in cv_kpts:
            # x, y!!
            # CONTINUE
            pt = np.array(cv_kpt.pt) + self.abs_adjustment
            cv_kpt.pt = tuple(pt)

    def create_cv_kpts(self, keypoints, scales, oris):
        keypoints = keypoints[0]
        kpts = []
        for i, kpt in enumerate(keypoints):
            x = kpt[0].item() + self.abs_adjustment[1]
            y = kpt[1].item() + self.abs_adjustment[0]
            size = scales[0, i].item()
            angle = oris[0, i].item()
            kp = cv.KeyPoint(x, y, size=size, angle=angle)
            kpts.append(kp)
        return kpts

    def detectAndCompute(self, img_np, mask):

        if len(img_np.shape) == 2:
            img_np = img_np[:, :, None]
        else:
            img_np = cv.cvtColor(img_np, cv.COLOR_BGR2GRAY)

        img_t = KU.image_to_tensor(img_np, False).float() / 255.
        ret_dict = self.dog({"image": img_t})
        cv_kpts = self.create_cv_kpts(ret_dict['keypoints'], ret_dict['scales'], ret_dict['oris'])
        cv_descs = ret_dict['descriptors'][0].T.numpy()
        return cv_kpts, cv_descs
