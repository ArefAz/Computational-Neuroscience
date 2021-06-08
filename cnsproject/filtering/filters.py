import numpy as np
import math

import torch


def make_gaussian(size, sig: float, center=None) -> np.ndarray:
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    gaussian = np.exp(
        -(((x - x0) ** 2) + ((y - y0) ** 2)) / (2 * sig ** 2)
    ) / math.sqrt(2 * math.pi)
    return gaussian


def dog_filter(
        size, sig1: float, sig2: float,
        is_off_center: bool = False
) -> np.ndarray:
    if sig2 <= sig1:
        raise ValueError('sig1 should be greater than sig2.')
    if size % 2 != 1:
        raise ValueError('Filter size should be odd.')
    g1 = make_gaussian(size, sig=sig1)
    g2 = make_gaussian(size, sig=sig2)
    dog: np.ndarray = g1 - g2
    dog -= dog.sum() / dog.size
    if is_off_center:
        dog *= -1
    return dog


def gabor_filter(
        k_size=111,
        sig: float = 10.,
        gamma: float = 1.2,
        lam: float = 10.,
        theta: float = 0.,
        is_off_center: bool = False
) -> np.ndarray:
    x0 = y0 = k_size // 2
    gabor = np.zeros((k_size, k_size))

    theta = math.radians(theta)
    for y in range(k_size):
        for x in range(k_size):
            px = x - x0
            py = y - y0

            X = math.cos(theta) * px + math.sin(theta) * py
            Y = -math.sin(theta) * px + math.cos(theta) * px

            cos = math.cos(2 * math.pi * X / lam)
            exp = math.exp(-(X ** 2 + gamma ** 2 * Y ** 2) / (2 * sig ** 2))
            gabor[x, y] = exp * cos

    # Make the filter zero-sum
    gabor -= gabor.sum() / gabor.size
    if is_off_center:
        gabor *= -1
    return gabor


def conv2d(
        image: np.ndarray,
        kernel: np.ndarray,
        padding='same',
        strides=(1, 1),
        output_tensor: bool = True,
        zero_one_norm: bool = False,
) -> np.ndarray:
    # Flip the kernel along x and y axis in order to perform convolution
    # instead of cross-correlation
    flipped_kernel = np.flipud(np.fliplr(kernel))
    x_ker, y_ker = flipped_kernel.shape
    x_input, y_input = image.shape[0:2]
    image_padded = image

    # Handle padding, 'valid' means no padding while 'same' preserves the input
    # shape by zero-padding.
    if padding == 'valid':
        output_h = int(math.ceil((y_input - y_ker + 1) / strides[1]))
        output_w = int(math.ceil((x_input - x_ker + 1) / strides[0]))
    elif padding == 'same':
        output_h = int(math.ceil(y_input / strides[1]))
        output_w = int(math.ceil(x_input / strides[0]))

        if y_input % strides[1] == 0:
            pad_h = max((y_ker - strides[1]), 0)
        else:
            pad_h = max(y_ker - (y_input % strides[1]), 0)
        if x_input % strides[0] == 0:
            pad_w = max((x_ker - strides[0]), 0)
        else:
            pad_w = max(x_ker - (x_input % strides[0]), 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        image_padded = np.zeros((x_input + pad_h, y_input + pad_w))
        image_padded[pad_top:-pad_bottom, pad_left:-pad_right] = image
    else:
        raise ValueError("padding should be set to either 'same' or 'valid'!")

    output = np.zeros((output_h, output_w))
    st_x = strides[0]
    st_y = strides[1]
    # Iterate over all pixels of the output and calculate its value
    # by using dot-product of the corresponding image patch and the filter
    for x in range(output_w):
        for y in range(output_h):
            patch = image_padded[y * st_y:y * st_y + y_ker,
                                 x * st_x:x * st_x + x_ker]
            output[y, x] = (flipped_kernel * patch).sum()

    if zero_one_norm:
        output -= output.min()
        output /= output.max()
    if output_tensor:
        output = torch.tensor(output, dtype=torch.float32)
    return output
