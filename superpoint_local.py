import torch
import sys
sys.path.append("./superpoint_forked")
from superpoint import SuperPointDescriptor
import cv2 as cv

class SuperPointDetector:

    def __init__(self, path=None, device: torch.device = torch.device('cpu')):
        if not path:
            path = "./superpoint_forked/superpoint_v1.pth"
        self.super_point = SuperPointDescriptor(path, device)

    def detect(self, img_np, mask=None):
        if len(img_np.shape) == 3:
            img_np = cv.cvtColor(img_np, cv.COLOR_RGB2GRAY)
        pts, _ = self.super_point.detectAndComputeGrey(img_np, mask)
        return pts
