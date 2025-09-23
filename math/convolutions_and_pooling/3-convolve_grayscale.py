#!/usr/bin/env python3
"""General grayscale convolution with padding and stride"""
import numpy as np

def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with custom padding and stride.
    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)
        padding: tuple of (ph, pw), 'same', or 'valid'
        stride: tuple of (sh, sw)
    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding
    if type(padding) is tuple:
        ph, pw = padding
        pad_top = ph // 2
        pad_bottom = ph - pad_top
        pad_left = pw // 2
        pad_right = pw - pad_left
    elif padding == 'same':
        # Compute total padding needed on height and width
        ph_total = max((h - 1) * sh + kh - h, 0)
        pw_total = max((w - 1) * sw + kw - w, 0)
        # Distribute padding to top/bottom and left/right
        pad_top = ph_total // 2
        pad_bottom = ph_total - pad_top
        pad_left = pw_total // 2
        pad_right = pw_total - pad_left
    elif padding == 'valid':
        pad_top = pad_bottom = pad_left = pad_right = 0
    else:
        raise ValueError("padding must be 'same', 'valid', or a tuple of (ph, pw)")

    # Pad images
    images_padded = np.pad(
        images,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant'
    )

    # Output dimensions
    out_h = ((h + pad_top + pad_bottom - kh) // sh) + 1
    out_w = ((w + pad_left + pad_right - kw) // sw) + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            i_start = i * sh
            j_start = j * sw
            region = images_padded[:, i_start:i_start+kh, j_start:j_start+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
