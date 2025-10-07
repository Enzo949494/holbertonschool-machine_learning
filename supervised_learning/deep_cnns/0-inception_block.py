#!/usr/bin/env python3
"""Inception block implementation"""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in Going Deeper with Convolutions (2014)
    
    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing F1, F3R, F3, F5R, F5, FPP
            F1: number of filters in the 1x1 convolution
            F3R: number of filters in the 1x1 convolution before the 3x3 convolution
            F3: number of filters in the 3x3 convolution
            F5R: number of filters in the 1x1 convolution before the 5x5 convolution
            F5: number of filters in the 5x5 convolution
            FPP: number of filters in the 1x1 convolution after the max pooling
    
    Returns:
        concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    
    # 1x1 convolution branch
    conv_1x1 = K.layers.Conv2D(F1, (1, 1), padding='same', activation='relu')(A_prev)
    
    # 1x1 -> 3x3 convolution branch
    conv_3x3_reduce = K.layers.Conv2D(F3R, (1, 1), padding='same', activation='relu')(A_prev)
    conv_3x3 = K.layers.Conv2D(F3, (3, 3), padding='same', activation='relu')(conv_3x3_reduce)
    
    # 1x1 -> 5x5 convolution branch
    conv_5x5_reduce = K.layers.Conv2D(F5R, (1, 1), padding='same', activation='relu')(A_prev)
    conv_5x5 = K.layers.Conv2D(F5, (5, 5), padding='same', activation='relu')(conv_5x5_reduce)
    
    # 3x3 max pooling -> 1x1 convolution branch
    max_pool = K.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(A_prev)
    pool_proj = K.layers.Conv2D(FPP, (1, 1), padding='same', activation='relu')(max_pool)
    
    # Concatenate all branches
    output = K.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj])
    
    return output