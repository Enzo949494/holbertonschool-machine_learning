#!/usr/bin/env python3
"""
Image data augmentation using horizontal flipping.

This module provides functions to perform horizontal flip augmentation
on images using TensorFlow.
"""

import tensorflow as tf


def flip_image(image):
    """
    Flip an image horizontally (left-right).

    Args:
        image: A tensor representing an image.

    Returns:
        A tensor with the image flipped horizontally.
    """
    return tf.image.flip_left_right(image)
