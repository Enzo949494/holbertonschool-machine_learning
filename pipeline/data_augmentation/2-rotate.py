#!/usr/bin/env python3
"""
Image data augmentation using rotation.

This module provides functions to perform rotation augmentation
on images using TensorFlow.
"""

import tensorflow as tf


def rotate_image(image):
    """
    Rotate an image by 90 degrees counterclockwise.

    Args:
        image: A tensor representing an image.

    Returns:
        A tensor with the image rotated 90 degrees counterclockwise.
    """
    return tf.image.rot90(image)
