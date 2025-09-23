#!/usr/bin/env python3
"""General grayscale convolution with padding and stride"""

import numpy as np

def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with custom padding and stride.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding and output size
    if type(padding) is tuple:
        ph, pw = padding
        pad_top = ph
        pad_bottom = ph
        pad_left = pw
        pad_right = pw
        out_h = ((h + 2 * ph - kh) // sh) + 1
        out_w = ((w + 2 * pw - kw) // sw) + 1
    elif padding == 'same':
        # Asymmetric padding for 'same'
        pad_h = max((h - 1) * sh + kh - h, 0)
        pad_w = max((w - 1) * sw + kw - w, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        out_h = ((h + pad_h - kh) // sh) + 1
        out_w = ((w + pad_w - kw) // sw) + 1
    elif padding == 'valid':
        # Pas de padding pour 'valid'
        images_padded = images
        out_h = ((h - kh) // sh) + 1
        out_w = ((w - kw) // sw) + 1
        pad_top = pad_bottom = pad_left = pad_right = 0
    else:
        raise ValueError("padding must be 'same', 'valid', or a tuple of (ph, pw)")

    if padding != 'valid':
        images_padded = np.pad(
            images,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant'
        )

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            i_start = i * sh
            j_start = j * sw
            region = images_padded[:, i_start:i_start+kh, j_start:j_start+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output