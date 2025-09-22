#!/usr/bin/env python3
"""same convolution"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w)
                containing multiple grayscale images
        kernel: numpy.ndarray with shape (kh, kw)
                containing the kernel for the convolution

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Determine padding amounts for height and width
    pad_top = kh // 2
    pad_bottom = kh - 1 - pad_top
    pad_left = kw // 2
    pad_right = kw - 1 - pad_left

    # Pad images with zeros (constant mode)
    images_padded = np.pad(
        images,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant'
    )

    # Prepare output array
    output = np.zeros((m, h, w))

    # Perform convolution: two for loops (vectorized over images dimension)
    for i in range(h):
        for j in range(w):
            region = images_padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
