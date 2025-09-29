#!/usr/bin/env python3
"""cnn"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer.
    
    Args:
        A_prev (np.ndarray): output from previous layer (m, h_prev, w_prev, c_prev)
        W (np.ndarray): kernels (filter weights) (kh, kw, c_prev, c_new)
        b (np.ndarray): biases (1, 1, 1, c_new)
        activation (function): activation function to apply
        padding (str): "same" or "valid"
        stride (tuple): (sh, sw) strides for height and width

    Returns:
        np.ndarray: output of convolutional layer (m, h_new, w_new, c_new)
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev_w, c_new = W.shape
    sh, sw = stride
    
    # Check channel dimension match
    if c_prev != c_prev_w:
        raise ValueError("Input channels and filter channels do not match.")
    
    # Compute padding width
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == "valid":
        ph, pw = 0, 0
    else:
        raise ValueError("padding must be 'same' or 'valid'")
    
    # Compute output dimensions
    h_new = int((h_prev + 2 * ph - kh) / sh) + 1
    w_new = int((w_prev + 2 * pw - kw) / sw) + 1
    
    # Pad input
    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )
    
    # Initialize output
    Z = np.zeros((m, h_new, w_new, c_new))
    
    # Perform convolution
    for i in range(m):  # iterate over examples
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    # Slice input
                    a_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    # Element-wise product and sum + bias
                    Z[i, h, w, c] = np.sum(a_slice * W[:, :, :, c]) + b[:, :, :, c]
    
    # Apply activation function
    A = activation(Z)
    
    return A