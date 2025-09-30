#!/usr/bin/env python3
"""pooling propagation"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer.

    Args:
    - A_prev (np.ndarray): shape (m, h_prev, w_prev, c_prev),
                           output of previous layer
    - kernel_shape (tuple): (kh, kw), size of the pooling kernel
    - stride (tuple): (sh, sw), strides for height and width
    - mode (str): 'max' or 'avg', pooling type

    Returns:
    - A (np.ndarray): output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1

    # Initialize output
    A = np.zeros((m, h_out, w_out, c_prev))

    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                for c in range(c_prev):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = A_prev[
                        i, vert_start:vert_end, horiz_start:horiz_end, c]

                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_slice)
                    elif mode == 'avg':
                        A[i, h, w, c] = np.mean(a_slice)

    return A
