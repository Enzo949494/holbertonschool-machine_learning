#!/usr/bin/env python3
"""
Image data augmentation using random cropping.

This module provides functions to perform random crop augmentation
on images using TensorFlow.
"""

import tensorflow as tf


def crop_image(image, size):
    """
    Crop an image randomly to a specified size.

    Args:
        image: A tensor representing an image.
        size: A tuple or list [height, width, channels] specifying the
              size of the cropped output.

    Returns:
        A tensor with the image randomly cropped to the specified size.
    """
    return tf.image.random_crop(image, size)
