import torch
import sys
sys.path.append("./superpoint_forked")
from superpoint import SuperPointDescriptor


class SuperPointDetector:

    def __init__(self, path=None, device: torch.device = torch.device('cpu')):
        if not path:
            path = "./superpoint_forked/superpoint_v1.pth"
        self.super_point = SuperPointDescriptor(path, device)

    def detect(self, img_np, mask=None):
        pts, _ = self.super_point.detectAndComputeGrey(img_np, mask)
        return pts
