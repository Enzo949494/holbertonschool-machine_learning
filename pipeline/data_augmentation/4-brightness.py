#!/usr/bin/env python3
"""
Image data augmentation using brightness adjustment.

This module provides functions to perform random brightness augmentation
on images using TensorFlow.
"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Adjust the brightness of an image randomly.

    Args:
        image: A tensor representing an image with values in [0, 1].
        max_delta: A float value representing the maximum change in brightness.
                   The brightness will be adjusted by a random amount in the
                   range [-max_delta, max_delta].

    Returns:
        A tensor with the image's brightness randomly adjusted by a value
        between -max_delta and max_delta.
    """
    return tf.image.random_brightness(image, max_delta)
