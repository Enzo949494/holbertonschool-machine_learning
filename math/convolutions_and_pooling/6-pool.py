#!/usr/bin/env python3
"""Pooling on images"""

import numpy as np

def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images: numpy.ndarray (m, h, w, c)
        kernel_shape: tuple (kh, kw)
        stride: tuple (sh, sw)
        mode: 'max' or 'avg'

    Returns:
        numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = ((h - kh) // sh) + 1
    out_w = ((w - kw) // sw) + 1

    output = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            i_start = i * sh
            j_start = j * sw
            region = images[:, i_start:i_start+kh, j_start:j_start+kw, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(region, axis=(1, 2))
            else:
                raise ValueError("mode must be 'max' or 'avg'")

    return output