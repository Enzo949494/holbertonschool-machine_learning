#!/usr/bin/env python3
"""
Image data augmentation using contrast adjustment.

This module provides functions to perform random contrast augmentation
on images using TensorFlow.
"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Adjust the contrast of an image randomly.

    Args:
        image: A tensor representing an image with values in [0, 1].
        lower: A float value representing the lower bound for the contrast
               adjustment factor.
        upper: A float value representing the upper bound for the contrast
               adjustment factor.

    Returns:
        A tensor with the image's contrast randomly adjusted between the
        specified lower and upper bounds.
    """
    return tf.image.random_contrast(image, lower, upper)
