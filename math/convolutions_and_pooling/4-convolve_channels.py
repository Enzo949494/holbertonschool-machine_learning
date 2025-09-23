#!/usr/bin/env python3
"""Convolution on images with channels"""

import numpy as np

def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Args:
        images: numpy.ndarray (m, h, w, c)
        kernel: numpy.ndarray (kh, kw, c)
        padding: tuple of (ph, pw), 'same', or 'valid'
        stride: tuple of (sh, sw)

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if type(padding) is tuple:
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        raise ValueError(
            "padding must be 'same', 'valid', or a tuple of (ph, pw)"
        )

    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    out_h = ((h + 2 * ph - kh) // sh) + 1
    out_w = ((w + 2 * pw - kw) // sw) + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            i_start = i * sh
            j_start = j * sw
            region = images_padded[:, i_start:i_start+kh, j_start:j_start+kw, :]
            # Sum over kh, kw, c for each image
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3))

    return output