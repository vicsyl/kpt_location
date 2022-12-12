import torch

import os
import torch.nn as nn
from kornia.filters.sobel import spatial_gradient3d
from kornia.utils import create_meshgrid3d
from typing import Tuple
from kornia.geometry.subpix.nms import nms3d
from kornia.utils.helpers import safe_solve_with_mask


def conv_quad_interp3d(
    input: torch.Tensor, strict_maxima_bonus: float = 10.0, eps: float = 1e-7
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the single iteration of quadratic interpolation of the extremum (max or min).

    Args:
        input: the given heatmap with shape :math:`(N, C, D_{in}, H_{in}, W_{in})`.
        strict_maxima_bonus: pixels, which are strict maxima will score (1 + strict_maxima_bonus) * value.
          This is needed for mimic behavior of strict NMS in classic local features
        eps: parameter to control the hessian matrix ill-condition number.

    Returns:
        the location and value per each 3x3x3 window which contains strict extremum, similar to one done is SIFT.
        :math:`(N, C, 3, D_{out}, H_{out}, W_{out})`, :math:`(N, C, D_{out}, H_{out}, W_{out})`,

        where

         .. math::
             D_{out} = \left\lfloor\frac{D_{in}  + 2 \times \text{padding}[0] -
             (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

         .. math::
             H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[1] -
             (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

         .. math::
             W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[2] -
             (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Examples:
        >>> input = torch.randn(20, 16, 3, 50, 32)
        >>> nms_coords, nms_val = conv_quad_interp3d(input, 1.0)
    """

    # coords or: [0, 0, 0, 3, 322, 354]
    # coords rot: [0, 0, 0, 3, 382, 322]

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 5:
        raise ValueError(f"Invalid input shape, we expect BxCxDxHxW. Got: {input.shape}")

    B, CH, D, H, W = input.shape
    grid_global: torch.Tensor = create_meshgrid3d(D, H, W, False, device=input.device).permute(0, 4, 1, 2, 3)
    grid_global = grid_global.to(input.dtype)

    # to determine the location we are solving system of linear equations Ax = b, where b is 1st order gradient
    # and A is Hessian matrix
    b: torch.Tensor = spatial_gradient3d(input, order=1, mode='diff')  #
    b_orig = b
    b = b.permute(0, 1, 3, 4, 5, 2).reshape(-1, 3, 1)
    A: torch.Tensor = spatial_gradient3d(input, order=2, mode='diff')
    A_orig = A
    # if Cfg.counter == 0:
    #     A_orig[:, :, 0, 1, 6, 457] = 10000
    # else:
    #     A_orig[:, :, 0, 1, 279, 6] = 10000

    A = A.permute(0, 1, 3, 4, 5, 2).reshape(-1, 6)

    # idx = torch.where(A[..., 0] == 10000)
    # print(f"idx: {idx}")


    dxx = A[..., 0]
    dyy = A[..., 1]
    dss = A[..., 2]
    dxy = 0.25 * A[..., 3]  # normalization to match OpenCV implementation
    dys = 0.25 * A[..., 4]  # normalization to match OpenCV implementation
    dxs = 0.25 * A[..., 5]  # normalization to match OpenCV implementation

    Hes = torch.stack([dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss], dim=-1).view(-1, 3, 3)

    # if not torch_version_geq(1, 10):
    #     # The following is needed to avoid singular cases
    #     Hes += torch.rand(Hes[0].size(), device=Hes.device).abs()[None] * eps

    nms_mask: torch.Tensor = nms3d(input, (3, 3, 3), True)
    x_solved: torch.Tensor = torch.zeros_like(b)

    x_solved_masked, _, solved_correctly = safe_solve_with_mask(b[nms_mask.view(-1)], Hes[nms_mask.view(-1)])


    # if Cfg.counter == 0:
    #     Hes[359376][0, 0] = 10000
    #     b[359376][0, 0] = 10000
    # else:
    #     Hes[488702][0, 0] = 10000
    #     b[488702][0, 0] = 10000
    bar = b[nms_mask.view(-1)]
    foo = Hes[nms_mask.view(-1)]
    # print(f"torch.where(bar == 10000): {torch.where(bar == 10000)}")
    # print(f"torch.where(foo == 10000): {torch.where(foo == 10000)}")
    if Cfg.counter == 0:
        print(f"foo[2]: {foo[2]}")
        print(f"bar[2]: {bar[2]}")
        print(f"x_solved_masked[2]: {x_solved_masked[2]}")
        # x_solved_masked[2, 0, 0] = 100
        masked_index = 2
    else:
        print(f"foo[131]: {foo[131]}")
        print(f"bar[131]: {bar[131]}")
        print(f"x_solved_masked[131]: {x_solved_masked[131]}")
        # x_solved_masked[131, 0, 0] = 100
        masked_index = 131
    print(f"torch.where(x_solved_masked[:, :, :] == 100): {torch.where(x_solved_masked[:, :, :] == 100)}")

    # print(f"bar[131, 0, 0]: {torch.where(bar == 10000)}")
    # print(f"torch.where(foo == 10000): {torch.where(foo == 10000)}")

    #  Kill those points, where we cannot solve
    new_nms_mask = nms_mask.masked_scatter(nms_mask, solved_correctly)

    print(f"x_solved != 0.0).sum(): {(x_solved != 0.0).sum()}")
    print(f"(x_solved != 0).sum(): {(x_solved != 0).sum()}")
    # FIXME this is the place !!!
    # original code
    # x_solved.masked_scatter_(new_nms_mask.view(-1, 1, 1), x_solved_masked[solved_correctly])
    # x_solved[torch.where(new_nms_mask.view(-1, 1, 1))[0][masked_index]] = x_solved_masked[solved_correctly][masked_index]
    x_solved[torch.where(new_nms_mask.view(-1, 1, 1))[0]] = x_solved_masked[solved_correctly]

    print(f"torch.where(x_solved[:, :, :] == 100): {torch.where(x_solved[:, :, :] == 100)}")

    print(f"new_nms_mask.sum(): {new_nms_mask.sum()}")
    print(f"x_solved != 0.0).sum(): {(x_solved != 0.0).sum()}")

    print(f"x_solved_masked[solved_correctly].sum(): {x_solved_masked[solved_correctly].sum()}")
    print(f"x_solved_masked[solved_correctly].abs().sum(): {x_solved_masked[solved_correctly].abs().sum()}")

    dx: torch.Tensor = -x_solved
    if Cfg.counter == 0:
        print(f"lin: hessian : {Hes[359376]}")
        print(f"b: {b[359376]}")
        print(f"dx : {dx[359376]}")
    else:
        print(f"lin: hessian : {Hes[488702]}")
        print(f"b: {b[488702]}")
        print(f"dx: {dx[488702]}")

    # Ignore ones, which are far from window center
    print(f"torch.where(dx[:, :, :] == -100): {torch.where(dx[:, :, :] == -100)}")
    mask1 = dx.abs().max(dim=1, keepdim=True)[0] > 0.7
    #mask1 = dx.abs().max(dim=1, keepdim=True)[0] > 100000
    dx.masked_fill_(mask1.expand_as(dx), 0)
    print(f"torch.where(dx[:, :, :] == -100): {torch.where(dx[:, :, :] == -100)}")

    dy: torch.Tensor = 0.5 * torch.bmm(b.permute(0, 2, 1), dx)
    y_max = input + dy.view(B, CH, D, H, W)
    if strict_maxima_bonus > 0:
        y_max += strict_maxima_bonus * new_nms_mask.to(input.dtype)

    print(f"torch.where(dx[:, :, :] == -100): {torch.where(dx[:, :, :] == -100)}")
    dx_res: torch.Tensor = dx.flip(1).reshape(B, CH, D, H, W, 3).permute(0, 1, 5, 2, 3, 4)
    print(f"torch.where(dx_res[:, :, :] == -100): {torch.where(dx_res[:, :, :] == -100)}")

    coords_max: torch.Tensor = grid_global.repeat(B, 1, 1, 1, 1).unsqueeze(1)

    # CONTINUE these coords with c_m and test

    if Cfg.counter == 0:
        # continue -> this is
        Cfg.nms_mask_or = nms_mask
        Cfg.new_nms_mask_or = new_nms_mask
        Cfg.dx_res = dx_res
        # coords_max.shape = torch.Size([1, 1, 3, 5, 481, 737])
        # coords_max[0, 0, 1, 1, 6, 279] = 279.0009
        # coords_max_sub = coords_max[0, 0, 1, 1, 6, 279].item()
        coords_max[0, 0, 1, 1, 6, 457] = 457.0009
        coords_max_sub = coords_max[0, 0, 1, 1, 6, 457].item()
        # dx_res_sub = dx_res[0, 0, 1, 1, 6, 279].item()
        dx_res_sub = dx_res[0, 0, :, 1, 6, 457]

        # input.shape = torch.Size([1, 1, 5, 481, 737])
        # input_sub = input[:, :, 0:3, 5:8, 278:281]
        # nms_mask_sub = nms_mask[:, :, 0:3, 5:8, 278:281]
        input_sub = input[:, :, 0:3, 5:8, 456:459]
        nms_mask_sub = nms_mask[:, :, 0:3, 5:8, 456:459]

        # A_sub = A_orig[:, :, :, 1, 6, 279]
        # b_sub = b_orig[:, :, :, 1, 6, 279]
        Hes_sub = A_orig[:, :, :, 1, 6, 457]
        A_sub = A_orig[:, :, :, 1, 6, 457]
        b_sub = b_orig[:, :, :, 1, 6, 457]

        dxx_lin = A_orig[0, 0, 0, 1, 6, 457]
        dyy_lin = A_orig[0, 0, 1, 1, 6, 457]
        dss_lin = A_orig[0, 0, 2, 1, 6, 457]
        dxy_lin = 0.25 * A_orig[0, 0, 3, 1, 6, 457]
        dys_lin = 0.25 * A_orig[0, 0, 4, 1, 6, 457]
        dxs_lin = 0.25 * A_orig[0, 0, 5, 1, 6, 457]
        Cfg.Hes_lin = torch.stack([dxx_lin, dxy_lin, dxs_lin, dxy_lin, dyy_lin, dys_lin, dxs_lin, dys_lin, dss_lin], dim=-1).view(-1, 3, 3)
        Cfg.b_lin = b_orig[:, :, :, 1, 6, 457].permute(0, 2, 1)

        # TODO: construct hessian and b, solve:
        #safe_solve_with_mask(B: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # test the result,
        # test the other result ....


        # coords_max_sub: 279.0
        # dx_res_sub: -0.0
        # input_sub: tensor([[[[[-0.0026, -0.0039, -0.0041],
        #            [-0.0025, -0.0030, -0.0028],
        #            [-0.0023, -0.0020, -0.0014]],
        #           [[-0.0027, -0.0032, -0.0028],
        #            [-0.0030, -0.0032, -0.0027],
        #            [-0.0026, -0.0025, -0.0017]],
        #           [[-0.0022, -0.0023, -0.0019],
        #            [-0.0023, -0.0024, -0.0019],
        #            [-0.0019, -0.0018, -0.0012]]]]])
        # A_sub: tensor([[[ 0.0008,  0.0008,  0.0011,  0.0011,  0.0014, -0.0008]]])
        # b_sub: tensor([[[0.0002, 0.0004, 0.0003]]])
    elif Cfg.counter == 1:
        # CONTINUE -> this is probably going to check out
        # check nms_mask against Cfg.nms_mask_or
        Cfg.nms_mask_or = torch.rot90(Cfg.nms_mask_or, 1, [3, 4])
        Cfg.new_nms_mask_or = torch.rot90(Cfg.new_nms_mask_or, 1, [3, 4])
        Cfg.dx_res = torch.rot90(Cfg.dx_res, 1, [4, 5])

        dxx_lin = A_orig[0, 0, 0, 1, 279, 6]
        dyy_lin = A_orig[0, 0, 1, 1, 279, 6]
        dss_lin = A_orig[0, 0, 2, 1, 279, 6]
        dxy_lin = 0.25 * A_orig[0, 0, 3, 1, 279, 6]
        dys_lin = 0.25 * A_orig[0, 0, 4, 1, 279, 6]
        dxs_lin = 0.25 * A_orig[0, 0, 5, 1, 279, 6]
        Hes_lin = torch.stack([dxx_lin, dxy_lin, dxs_lin, dxy_lin, dyy_lin, dys_lin, dxs_lin, dys_lin, dss_lin], dim=-1).view(-1, 3, 3)
        b_lin = b_orig[:, :, :, 1, 279, 6].permute(0, 2, 1)

        x_solved_masked_lin_or, _, solved_correctly_lin_or = safe_solve_with_mask(Cfg.b_lin, Cfg.Hes_lin)
        print(f"Original Hessian: {Cfg.Hes_lin}")
        print(f"Original b: {Cfg.b_lin}")
        print(f"Original x: {x_solved_masked_lin_or}")
        x_solved_masked_lin, _, solved_correctly_lin = safe_solve_with_mask(b_lin, Hes_lin)
        print(f"Rot Hessian: {Hes_lin}")
        print(f"Rot b: {b_lin}")
        print(f"Rot x: {x_solved_masked_lin}")


        print(f"nms_mask == Cfg.nms_mask_or: {(nms_mask == Cfg.nms_mask_or).sum()}")
        print(f"nms_mask == Cfg.nms_mask_or: {(nms_mask == Cfg.nms_mask_or).sum()}")

        print(f"nms_mask == Cfg.nms_mask_or: {(nms_mask == Cfg.nms_mask_or).sum()}")
        print(f"nms_mask != Cfg.nms_mask_or: {(nms_mask != Cfg.nms_mask_or).sum()}")
        print(f"(new_nms_mask == Cfg.new_nms_mask_or).sum(): {(new_nms_mask == Cfg.new_nms_mask_or).sum()}")
        print(f"(new_nms_mask != Cfg.new_nms_mask_or).sum(): {(new_nms_mask != Cfg.new_nms_mask_or).sum()}")
        print(f"(dx_res == Cfg.dx_res).sum(): {(dx_res == Cfg.dx_res).sum()}")
        print(f"(dx_res != Cfg.dx_res).sum(): {(dx_res != Cfg.dx_res).sum()}")
        print(f"torch.unique(dx_res): {torch.unique(dx_res)}")
        print(f"torch.unique(Cfg.dx_res): {torch.unique(Cfg.dx_res)}")

        # coords_max_sub: 279.0
        # dx_res_sub: -0.6956359148025513
        # input_sub: tensor([[[[[0.0085, 0.0097, 0.0082],
        #            [0.0077, 0.0090, 0.0077],
        #            [0.0070, 0.0079, 0.0067]],
        #           [[0.0089, 0.0093, 0.0080],
        #            [0.0094, 0.0101, 0.0086],
        #            [0.0087, 0.0095, 0.0080]],
        #           [[0.0087, 0.0086, 0.0075],
        #            [0.0095, 0.0095, 0.0079],
        #            [0.0089, 0.0088, 0.0070]]]]])
        # A_sub: tensor([[[-0.0022, -0.0015, -0.0018,  0.0001, -0.0019,  0.0016]]])
        # b_sub: tensor([[[-0.0004,  0.0001,  0.0002]]])
        coords_max_sub = coords_max[0, 0, 2, 1, 279, 6].item()
        dx_res_sub = dx_res[0, 0, :, 1, 279, 6]
        # 457 = 737 - 1 - 279
        # input_sub = input[:, :, 0:3, 456:459, 5:8]
        # nms_mask_sub = nms_mask[:, :, 0:3, 456:459, 5:8]
        input_sub = input[:, :, 0:3, 278:281, 5:8]
        nms_mask_sub = nms_mask[:, :, 0:3, 278:281, 5:8]

        # A_sub = A_orig[:, :, :, 1, 457, 6]
        # b_sub = b_orig[:, :, :, 1, 457, 6]
        A_sub = A_orig[:, :, :, 1, 279, 6]
        #Hes_sub = Hes_orig[:, :, :, 1, 279, 6]
        b_sub = b_orig[:, :, :, 1, 279, 6]
    else:
        coords_max_sub = "foo"
        dx_res_sub = "bar"
        input_sub = "bar"
        A_sub = "bar"
        b_sub = "bar"
        nms_mask_sub = "bar"
        A_sub = "bar"
        b_sub = "bar"


    print(f"new_nms_mask")
    print(f"coords_max_sub: {coords_max_sub}")
    print(f"coords_max.shape: {coords_max.shape}")
    print(f"dx_res_sub: {dx_res_sub}")
    print(f"input_sub: {input_sub}")
    #print(f"nms_mask_sub: {nms_mask_sub}")
    print(f"nms_mask.sum(): {nms_mask.sum()}")
    print(f"input.shape: {input.shape}")
    print(f"A_sub: {A_sub}")
    print(f"b_sub: {b_sub}")
    print(f"torch.where(x_solved == 0.6956359148025513): {torch.where(x_solved == 0.6956359148025513)}")
    print(f"torch.where(dx == -0.6956359148025513): {torch.where(dx == -0.6956359148025513)}")
    print(f"torch.where(dx_res == -0.6956359148025513): {torch.where(dx_res == -0.6956359148025513)}")
    print(f"torch.where(dx == 0.6956359148025513): {torch.where(dx == 0.6956359148025513)}")
    print(f"torch.where(dx_res == 0.6956359148025513): {torch.where(dx_res == 0.6956359148025513)}")

    coords_max = coords_max + dx_res
    print(f"dx_res.abs().max(): {dx_res.abs().max()}")

    return coords_max, y_max


class ConvQuadInterp3d(nn.Module):
    r"""Calculate soft argmax 3d per window.

    See :func:`~kornia.geometry.subpix.conv_quad_interp3d` for details.
    """

    def __init__(self, strict_maxima_bonus: float = 10.0, eps: float = 1e-7) -> None:
        super().__init__()
        self.strict_maxima_bonus = strict_maxima_bonus
        self.eps = eps
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(' + 'strict_maxima_bonus=' + str(self.strict_maxima_bonus) + ')'

    def forward(self, x: torch.Tensor):  # type: ignore
        return conv_quad_interp3d(x, self.strict_maxima_bonus, self.eps)


def load(fn, dir="work/scale_space"):
    return torch.load(f"{dir}/{fn}")


def compare_rot(original, rot, levels, axes=[3, 4]):

    for i in range(levels):
        rot_s = torch.rot90(rot[i], 3, axes)
        diff = original[i][0, 0] - rot_s[0, 0]
        values, counts = torch.unique(diff, return_counts=True)
        print(f"max: {diff.abs().max()}")
        print(values)
        # print(counts)


def compare(original, rot, levels):

    for i in range(levels):
        diff = original[i] - rot[i]
        values, counts = torch.unique(diff, return_counts=True)
        print(f"max: {diff.abs().max()}")
        print(values)
        # print(counts)


def run_nms(oct_resp_l, levels):

    num_feats = 8000
    coord_maxs = []
    response_maxs = []
    resp_flat_bests = []
    max_coords_bests = []

    for oct_resp in oct_resp_l[:levels]:

        nms_module = ConvQuadInterp3d(10)

        coord_max, response_max = nms_module(oct_resp)

        if Cfg.counter == 0:
            c_m = coord_max[0, 0, 1, 1, 6, 279].item()
            #test, _ = nms_module(oct_resp[:, :, 0:3, 5:8, 278:281])
        elif Cfg.counter == 1:
            c_m = coord_max[0, 0, 2, 1, 279, 6].item()
            #test, _ = nms_module(oct_resp[:, :, 0:3, 278:281, 5:8])
        else:
            c_m = "foo"
        test = "bar"
        print(f"c_m: {c_m}")
        print(f"test: {test}")

        minima_are_also_good = False # normally yes, though
        if minima_are_also_good:
            coord_min, response_min = nms_module(-oct_resp)
            take_min_mask = (response_min > response_max).to(response_max.dtype)
            response_max = response_min * take_min_mask + (1 - take_min_mask) * response_max
            coord_max = coord_min * take_min_mask.unsqueeze(2) + (1 - take_min_mask.unsqueeze(2)) * coord_max

        # if self.compensate_nms is not None and self.compensate_nms != 0:
        #     assert self.compensate_nms == 3
        #     oct_resp = torch.rot90(oct_resp, 4 - self.compensate_nms, [3, 4])
        #     coord_temp = torch.clone(coord_max[:, :, 2, :, :])
        #     coord_max[:, :, 2, :, :] = torch.clone(coord_max[:, :, 1, :, :])
        #     coord_max[:, :, 1, :, :] = coord_temp
        #     coord_max[:, :, 2] = torch.clone(torch.flip(coord_max[:, :, 2], [4]))
        #     coord_max = torch.clone(torch.rot90(coord_max, 1, [4, 5]))
        #     response_max = torch.clone(torch.rot90(response_max, 1, [3, 4]))
        #     # original to rotate by rot 4 - self.compensate_nms

        # Now, lets crop out some small responses
        # responses_flatten = response_max.view(response_max.size(0), -1)  # [B, N]
        responses_flatten = response_max.reshape(response_max.size(0), -1)  # [B, N]
        # max_coords_flatten = coord_max.view(response_max.size(0), 3, -1).permute(0, 2, 1)  # [B, N, 3]
        max_coords_flatten = coord_max.reshape(response_max.size(0), 3, -1).permute(0, 2, 1)  # [B, N, 3]

        if responses_flatten.size(1) > num_feats:
            resp_flat_best, idxs = torch.topk(responses_flatten, k=num_feats, dim=1)
            max_coords_best = torch.gather(max_coords_flatten, 1, idxs.unsqueeze(-1).repeat(1, 1, 3))
        else:
            resp_flat_best = responses_flatten
            max_coords_best = max_coords_flatten
        coord_maxs.append(coord_max)
        response_maxs.append(response_max)
        resp_flat_bests.append(resp_flat_best)
        max_coords_bests.append(max_coords_best)

    return coord_maxs, response_maxs, resp_flat_bests, max_coords_bests

class Cfg:
    counter = 0
    nms_mask_or = None
    new_nms_mask_or = None
    dx_res = None
    Hes_real = None
    Hes_lin = None
    b_lib = None


def debug_nms():
    levels = 1
    print("oct_resp_comp_")
    oct_resp_comp_or = [load(f"oct_resp_comp_0_{i}", dir="work/scale_space_nms") for i in range(levels)]
    oct_resp_comp_1 = [load(f"oct_resp_comp_1_{i}", dir="work/scale_space_nms") for i in range(levels)]
    compare_rot(oct_resp_comp_or, oct_resp_comp_1, levels)
    coord_maxs_or, response_maxs_or, resp_flat_bests_or, max_coords_bests_or = run_nms(oct_resp_comp_or, levels)
    Cfg.counter += 1
    coord_maxs_rot, response_maxs_rot, resp_flat_bests_rot, max_coords_bests_rot = run_nms(oct_resp_comp_1, levels)
    Cfg.counter += 1

    # or_or = coord_maxs_or[0][0, 0, 2, 1, 6, 457]
    # sub_or = torch.clone(oct_resp_comp_or[0][:, :, 0:3, 5:8, 278:281])
    # # tensor([[[[[-0.0026, -0.0039, -0.0041],
    # #            [-0.0025, -0.0030, -0.0028],
    # #            [-0.0023, -0.0020, -0.0014]],
    # #
    # #           [[-0.0027, -0.0032, -0.0028],
    # #            [-0.0030, -0.0032, -0.0027],
    # #            [-0.0026, -0.0025, -0.0017]],
    # #
    # #           [[-0.0022, -0.0023, -0.0019],
    # #            [-0.0023, -0.0024, -0.0019],
    # #            [-0.0019, -0.0018, -0.0012]]]]])
    # #
    # coord_maxs_or_sub, response_maxs_or_sub, _, _ = run_nms([sub_or], 1)
    # # rot2 = coord_maxs_rot[i][0, 0][2, 1, 279, 6]
    # sub_rot = torch.clone(oct_resp_comp_1[0][:, :, 0:3, 278:281, 5:8])
    # # tensor([[[[[0.0085, 0.0097, 0.0082],
    # #            [0.0077, 0.0090, 0.0077],
    # #            [0.0070, 0.0079, 0.0067]],
    # #
    # #           [[0.0089, 0.0093, 0.0080],
    # #            [0.0094, 0.0101, 0.0086],
    # #            [0.0087, 0.0095, 0.0080]],
    # #
    # #           [[0.0087, 0.0086, 0.0075],
    # #            [0.0095, 0.0095, 0.0079],
    # #            [0.0089, 0.0088, 0.0070]]]]])
    # #
    # coord_maxs_rot_sub, response_maxs_rot_sub, _, _ = run_nms([sub_rot], 1)

    print("     response max")
    for i in range(levels):
        response_maxs_rot[i] = torch.rot90(response_maxs_rot[i], 3, [3, 4])
        diff = response_maxs_rot[i][0, 0] - response_maxs_or[i][0, 0]
        values, counts = torch.unique(diff, return_counts=True)
        print(f"max: {diff.abs().max()}")
        print(f"values: {values}")
        print(f"counts: {counts}")
        # print(counts)


    # or_or = coord_maxs_or[0][0, 0, 0, 3, 382, 322]
    # w = 481

    # w = 737
    # or_or = coord_maxs_or[0][0, 0, 0, 3, 322, w - 1 - 382].item()
    # print(f"coords or: {[0, 0, 0, 3, 322, w - 1 - 382]}")

    # w = 737
    # or_or = coord_maxs_or[0][0, 0, 2, 3, 327, w - 1 - 14].item()
    # print(f"coords or: {[0, 0, 2, 3, 327, w - 1 - 14]}")

    w = 737
    or_or = coord_maxs_or[0][0, 0, 1, 1, 6, 279].item()
    print(f"coords or: {[0, 0, 1, 1, 6, 279]}")
    print(f"or_or: {or_or}")

    print("     coord max")
    for i in range(levels):

        coord_temp = torch.clone(coord_maxs_or[i][:, :, 2, :, :])
        coord_maxs_or[i][:, :, 2, :, :] = torch.clone(coord_maxs_or[i][:, :, 1, :, :])
        coord_maxs_or[i][:, :, 1, :, :] = coord_temp
        coord_maxs_or[i] = torch.rot90(coord_maxs_or[i], 1, [4, 5])
        coord_maxs_or[i][:, :, 2] = torch.clone(torch.flip(coord_maxs_or[i][:, :, 2], [4]))
        coord_maxs_or[i][:, :, 2] = 736 - coord_maxs_or[i][:, :, 2]

    #compare(coord_max_or, coord_max_1, levels, axes=[4, 5])
    for i in range(levels):
        # coord_maxs_or[i][:, :, 2] = 736 - coord_maxs_or[i][:, :, 2]
        # s = coord_max_or[i][0, 0] + coord_max_1[i][0, 0]
        diff = coord_maxs_or[i][0, 0] - coord_maxs_rot[i][0, 0]
        diff_f = coord_maxs_or[i][0, 0] + coord_maxs_rot[i][0, 0]
        # diff[:, :, :, :].abs().max(dim=0).values.max(dim=1).values.max(dim=1).values
        values, counts = torch.unique(diff, return_counts=True)
        print(f"values: {values}")
        print(f"counts: {counts}")
        diff_a = diff[2:].abs()
        print(f"max: {diff_a.max()}")

        coords_r = torch.where(diff_a.max() == diff_a)
        # coords_r = torch.where(diff_a.max() <= 0.5)
        # coords_rot = [i.item() for i in coords_r]
        # print(f"coords rot: {coords_rot}")

        # dim0=0
        # rot0 = coord_maxs_rot[i][0, 0][0, 3, 382, 322]
        # dim0=1
        # rot1 = coord_maxs_rot[i][0, 0][1, 3, 14, 327]
        # dim0=2
        rot2 = coord_maxs_rot[i][0, 0][2, 1, 279, 6]
        rot = rot2

        print(f"rot: {rot}")
        # rot2 = coord_maxs_rot[i][0, 0][coords]

        # or1 = coord_maxs_or[i][0, 0][0, 3, 382, 322]
        # or1 = coord_maxs_or[i][0, 0][1, 3, 14, 327]
        ## or_or = coord_maxs_or[0][0, 0, 1, 1, 6, 279].item()
        or1 = coord_maxs_or[i][0, 0][2, 1, 279, 6]
        print(f"or1: {or1}")

        print(f"or_or: {or_or}")

        diff_1 = rot - or1
        diff_2 = rot - or_or
        print(f"diff1: {diff_1}")
        print(f"diff2: {diff_2}")

        # or1 = coord_maxs_or[i][0, 0][coords]

        print(f"out of {torch.numel(diff)} ...")
        print(f" == 0: {(diff == 0).sum()}")
        print(f" > 0.1: {(diff_a > 0.1).sum()}")
        print(f" > 0.2: {(diff_a > 0.2).sum()}")
        print(f" > 0.5: {(diff_a > 0.5).sum()}")
        print(f"values: {values}")
        print(f"counts: {counts}")
        pass
        # print(counts)


def inspect():

    levels = 4
    oct_or = [load(f"oct_resp_0_{i}") for i in range(levels)]
    oct_1 = [load(f"oct_resp_1_{i}") for i in range(levels)]
    print("octave response")
    compare_rot(oct_or, oct_1, levels)
    # for i in range(levels):
    #     oct_1[i] = torch.rot90(oct_1[i], 3, [3, 4])
    #     diff = oct_or[i][0, 0] - oct_1[i][0, 0]
    #     values, counts = torch.unique(diff, return_counts=True)
    #     print(f"max: {diff.abs().max()}")
    #     print(values)
    #     # print(counts)

    print("oct_resp_comp_")
    oct_resp_comp_or = [load(f"oct_resp_comp_0_{i}") for i in range(levels)]
    oct_resp_comp_1 = [load(f"oct_resp_comp_1_{i}") for i in range(levels)]
    compare_rot(oct_resp_comp_or, oct_resp_comp_1, levels)
    # for i in range(levels):
    #     diff = oct_resp_comp_or[i][0, 0] - oct_resp_comp_1[i][0, 0]
    #     values, counts = torch.unique(diff, return_counts=True)
    #     print(f"max: {diff.abs().max()}")
    #     print(values)
    #     # print(counts)

    # print("coord_max_comp_")
    # coord_max_comp_or = [load(f"coord_max_comp_0_{i}") for i in range(levels)]
    # coord_max_comp_1 = [load(f"coord_max_comp_1_{i}") for i in range(levels)]
    # for i in range(levels):
    #     diff = coord_max_comp_or[i][0, 0] - coord_max_comp_1[i][0, 0]
    #     values, counts = torch.unique(diff, return_counts=True)
    #     print(f"max: {diff.abs().max()}")
    #     print(values)
    #     # print(counts)

    print("     coord max")
    coord_max_or = [load(f"coord_max_0_{i}") for i in range(levels)]
    coord_max_1 = [load(f"coord_max_1_{i}") for i in range(levels)]
    # for i in range(levels):
    #     coord_temp = torch.clone(coord_max_1[i][:, :, 2, :, :])
    #     coord_max_1[i][:, :, 2, :, :] = torch.clone(coord_max_1[i][:, :, 1, :, :])
    #     coord_max_1[i][:, :, 1, :, :] = coord_temp
    #     # coord_max_1[i][0, 0, 1] = torch.flip(coord_max_1[i][0, 0, 1], [1])
    #     coord_max_1[i][:, :, 1] = torch.flip(coord_max_1[i][:, :, 1], [3])

    # for i in range(levels):
    #     coord_max_1[i] = torch.rot90(coord_max_1[i], 3, [4, 5])
    #     # s = coord_max_or[i][0, 0] + coord_max_1[i][0, 0]
    #     diff = coord_max_or[i][0, 0] - coord_max_1[i][0, 0]
    #     values, counts = torch.unique(diff, return_counts=True)
    #     print(f"max: {diff.abs().max()}")
    #     print(values)
    #     # print(counts)

    for i in range(levels):
        coord_temp = torch.clone(coord_max_or[i][:, :, 2, :, :])
        coord_max_or[i][:, :, 2, :, :] = torch.clone(coord_max_or[i][:, :, 1, :, :])
        coord_max_or[i][:, :, 1, :, :] = coord_temp
        # coord_max_or[i][0, 0, 1] = torch.flip(coord_max_or[i][0, 0, 1], [1])
        coord_max_or[i][:, :, 2] = torch.flip(coord_max_or[i][:, :, 2], [4])

    #compare(coord_max_or, coord_max_1, levels, axes=[4, 5])
    for i in range(levels):
        coord_max_or[i] = torch.rot90(coord_max_or[i], 1, [4, 5])
        # s = coord_max_or[i][0, 0] + coord_max_1[i][0, 0]
        diff = coord_max_or[i][0, 0] - coord_max_1[i][0, 0]
        # diff[:, :, :, :].abs().max(dim=0).values.max(dim=1).values.max(dim=1).values
        values, counts = torch.unique(diff, return_counts=True)
        diff_a = diff.abs()
        print(f"max: {diff_a.max()}")
        print(f"out of {torch.numel(diff)} ...")
        print(f" == 0: {(diff == 0).sum()}")
        print(f" > 0.1: {(diff_a > 0.1).sum()}")
        print(f" > 0.2: {(diff_a > 0.2).sum()}")
        print(f" > 0.5: {(diff_a > 0.5).sum()}")
        print(f"values: {values}")
        print(f"counts: {counts}")
        # print(counts)

    print("response max")
    response_max_or = [load(f"response_max_0_{i}") for i in range(levels)]
    response_max_1 = [load(f"response_max_1_{i}") for i in range(levels)]
    # compare(response_max_or, response_max_1, levels)
    for i in range(levels):
        response_max_1[i] = torch.rot90(response_max_1[i], 3, [3, 4])
        diff = response_max_1[i][0, 0] - response_max_or[i][0, 0]
        values, counts = torch.unique(diff, return_counts=True)
        print(f"max: {diff.abs().max()}")
        print(values)
        # print(counts)

    print("responses_flatten_0_")
    responses_flatten_or = [load(f"responses_flatten_0_{i}") for i in range(levels)]
    responses_flatten_1 = [load(f"responses_flatten_1_{i}") for i in range(levels)]
    compare(responses_flatten_or, responses_flatten_1, levels)

    print("max_coords_flatten_or")
    max_coords_flatten_or = [load(f"max_coords_flatten_0_{i}") for i in range(levels)]
    max_coords_flatten_1 = [load(f"max_coords_flatten_1_{i}") for i in range(levels)]
    compare(max_coords_flatten_or, max_coords_flatten_1, levels)

    print("resp_flat_best")
    resp_flat_best_or = [load(f"resp_flat_best_0_{i}") for i in range(levels)]
    resp_flat_best_1 = [load(f"resp_flat_best_1_{i}") for i in range(levels)]
    compare(resp_flat_best_or, resp_flat_best_1, levels)

    print("max_coords_best")
    max_coords_best_or = [load(f"max_coords_best_0_{i}") for i in range(levels)]
    max_coords_best_1 = [load(f"max_coords_best_1_{i}") for i in range(levels)]
    compare(max_coords_best_or, max_coords_best_1, levels)

    print("current_lafs_final")
    current_lafs_or = [load(f"current_lafs_final_0_{i}") for i in range(levels)]
    current_lafs_1 = [load(f"current_lafs_final_1_{i}") for i in range(levels)]
    compare(current_lafs_or, current_lafs_1, levels)

    print("resp_flat_best_1")
    resp_flat_best_or = [load(f"resp_flat_best_final_0_{i}") for i in range(levels)]
    resp_flat_best_1 = [load(f"resp_flat_best_final_1_{i}") for i in range(levels)]
    compare(resp_flat_best_or, resp_flat_best_1, levels)

    print("current_lafs comp")
    current_lafs_comp_or = [load(f"current_lafs_comp_final_0_{i}") for i in range(levels)]
    current_lafs_comp__1 = [load(f"current_lafs_comp_final_1_{i}") for i in range(levels)]
    compare(current_lafs_comp_or, current_lafs_comp__1, levels)
#
#
# torch.save(current_lafs, f"work/scale_space/current_lafs_final_{ScaleSpaceDetector.counter}_{oct_idx}")
# torch.save(resp_flat_best, f"work/scale_space/resp_flat_best_final_{ScaleSpaceDetector.counter}_{oct_idx}")
# torch.save(current_lafs, f"work/scale_space/current_lafs_comp_final_{ScaleSpaceDetector.counter}_{oct_idx}")

if __name__ == "__main__":
    # inspect()
    debug_nms()
