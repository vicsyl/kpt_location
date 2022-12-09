import torch

import os


def load(fn):
    return torch.load(f"work/scale_space/{fn}")


def compare_rot(original, rot, levels, axes=[3, 4]):

    for i in range(levels):
        rot[i] = torch.rot90(rot[i], 3, axes)
        diff = original[i][0, 0] - rot[i][0, 0]
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

    print("coord_max_comp_")
    coord_max_comp_or = [load(f"coord_max_comp_0_{i}") for i in range(levels)]
    coord_max_comp_1 = [load(f"coord_max_comp_1_{i}") for i in range(levels)]
    for i in range(levels):
        diff = coord_max_comp_or[i][0, 0] - coord_max_comp_1[i][0, 0]
        values, counts = torch.unique(diff, return_counts=True)
        print(f"max: {diff.abs().max()}")
        print(values)
        # print(counts)

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


    # disable for now!!
    # for i in range(levels):
    #     coord_temp = torch.clone(coord_max_or[i][:, :, 2, :, :])
    #     coord_max_or[i][:, :, 2, :, :] = torch.clone(coord_max_or[i][:, :, 1, :, :])
    #     coord_max_or[i][:, :, 1, :, :] = coord_temp
    #     # coord_max_or[i][0, 0, 1] = torch.flip(coord_max_or[i][0, 0, 1], [1])
    #     coord_max_or[i][:, :, 2] = torch.flip(coord_max_or[i][:, :, 2], [4])
    #
    # #compare(coord_max_or, coord_max_1, levels, axes=[4, 5])
    # for i in range(levels):
    #     coord_max_or[i] = torch.rot90(coord_max_or[i], 1, [4, 5])
    #     # s = coord_max_or[i][0, 0] + coord_max_1[i][0, 0]
    #     diff = coord_max_or[i][0, 0] - coord_max_1[i][0, 0]
    #     values, counts = torch.unique(diff, return_counts=True)
    #     print(f"max: {diff.abs().max()}")
    #     print(values)
    #     # print(counts)


    print("response max")
    response_max_or = [load(f"response_max_0_{i}") for i in range(levels)]
    response_max_1 = [load(f"response_max_1_{i}") for i in range(levels)]
    # compare(response_max_or, response_max_1, levels)
    for i in range(levels):
        response_max_1[i] = torch.rot90(response_max_1[i], 3, [3, 4])
        diff = response_max_or[i][0, 0] - response_max_or[i][0, 0]
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
    inspect()
