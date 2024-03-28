import torch


def diff_x(input, r):
    assert input.dim() == 4

    left = input[:, :, r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
    right = input[:, :, -1:] - input[:, :, -2 * r - 1: -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output


def diff_y(input, r):
    assert input.dim() == 4

    left = input[:, :, :, r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
    right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1: -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output


def box_filter(x, r):
    assert x.dim() == 4

    return diff_y(torch.cumsum(diff_x(torch.cumsum(x, dim=2), r), dim=3), r)


def guided_filter(x: torch.Tensor, y: torch.Tensor, r, eps=1e-8):
    assert x.dim() == 4 and y.dim() == 4

    # data format (always NHWC in PyTorch)
    # No need for transpose

    # shape check
    x_shape = x.shape
    y_shape = y.shape

    assert x_shape[0] == y_shape[0], "Batch sizes must match"
    assert x_shape[2:] == y_shape[2:], "Input and guidance image must have the same spatial dimensions"
    assert (x_shape[2] > 2 * r + 1) and (x_shape[3] > 2 * r + 1), "Window size must be smaller than image dimensions"

    # N
    N = box_filter(torch.ones(
        (1, 1, x_shape[2], x_shape[3]), dtype=x.dtype), r)

    # mean_x
    mean_x = box_filter(x, r) / N
    # mean_y
    mean_y = box_filter(y, r) / N
    # cov_xy
    cov_xy = box_filter(x * y, r) / N - mean_x * mean_y
    # var_x
    var_x = box_filter(x * x, r) / N - mean_x * mean_x

    # A
    A = cov_xy / (var_x + eps)
    # b
    b = mean_y - A * mean_x

    mean_A = box_filter(A, r) / N
    mean_b = box_filter(b, r) / N

    output = mean_A * x + mean_b

    return output
