#!/usr/bin/env python3
"""convolution with custom padding"""

import numpy as np

def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)
        padding: tuple of (ph, pw)

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad images with zeros
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Output dimensions
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                images_padded[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )

    return output