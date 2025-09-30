#!/usr/bin/env python3
"""backpropagation cnn"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs backpropagation over a convolutional layer.

    Args:
        dZ (np.ndarray): Gradient of the cost
                         with respect to the output of the conv layer.
        A_prev (np.ndarray): Output activations of the previous layer.
        W (np.ndarray): Filters for the convolution.
        b (np.ndarray): Biases for the convolution.
        padding (str): Type of padding ("same" or "valid").
        stride (tuple): Stride for the convolution.

    Returns:
        dA_prev (np.ndarray): Gradient with respect to the previous layer.
        dW (np.ndarray): Gradient with respect to the filters.
        db (np.ndarray): Gradient with respect to the biases.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    m, h_new, w_new, c_new = dZ.shape
    sh, sw = stride

    if padding == 'same':
        pad_h = ((h_prev - 1) * sh - h_prev + kh + 1) // 2
        pad_w = ((w_prev - 1) * sw - w_prev + kw + 1) // 2
    elif padding == 'valid':
        pad_h, pad_w = 0, 0

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant'
    )
    dA_prev_pad = np.pad(
        dA_prev,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant'
    )

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = a_prev_pad[
                        vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window & the filter's parameters
                    da_prev_pad[
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        :
                    ] += (
                        W[:, :, :, c] * dZ[i, h, w, c]
                    )
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        if padding == 'same':
            dA_prev[i, :, :, :] = da_prev_pad[pad_h:-pad_h, pad_w:-pad_w, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad

    return dA_prev, dW, db
