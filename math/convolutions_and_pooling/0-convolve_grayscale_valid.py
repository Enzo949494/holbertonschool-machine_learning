#!/usr/bin/env python3
"""valid convolution"""

import numpy as np

def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple grayscale images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel for the convolution

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    output_h = h - kh + 1
    output_w = w - kw + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            # images[:, i:i+kh, j:j+kw] shape: (m, kh, kw)
            # kernel shape: (kh, kw)
            # element-wise multiply and sum over axes 1 and 2
            output[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return output