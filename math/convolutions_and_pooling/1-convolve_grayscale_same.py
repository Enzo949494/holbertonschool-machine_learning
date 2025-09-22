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

    # Calculate padding for height and width
    pad_h = (kh - 1) // 2
    pad_h_extra = (kh - 1) % 2
    pad_w = (kw - 1) // 2
    pad_w_extra = (kw - 1) % 2

    # Pad images with zeros
    images_padded = np.pad(
        images,
        ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)),
        mode='constant'
    )

    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(
                images_padded[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )

    return output