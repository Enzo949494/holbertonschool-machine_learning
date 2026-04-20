#!/usr/bin/env python3
"""
Image data augmentation using hue adjustment.

This module provides functions to perform hue augmentation
on images using TensorFlow.
"""

import tensorflow as tf


def change_hue(image, delta):
    """
    Adjust the hue of an image.

    Args:
        image: A tensor representing an RGB image with values in [0, 1].
        delta: A float value in the range [-0.5, 0.5] representing the
               amount to rotate the hue by. Positive values rotate the hue
               counterclockwise, negative values rotate it clockwise.

    Returns:
        A tensor with the image's hue adjusted by the specified delta value.
    """
    return tf.image.adjust_hue(image, delta)
