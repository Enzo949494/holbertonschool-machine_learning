#!/usr/bin/env python3
"""
Module for building a convolutional Generative Adversarial Network (GAN).

This module provides functionality to create a generator and discriminator
for a GAN architecture designed to generate 16x16 face images.
"""
import tensorflow as tf
from tensorflow import keras


def convolutional_GenDiscr():
    """
    Create a convolutional GAN with generator and discriminator.

    tuple: A tuple containing:
        - generator (keras.Model): Neural network generate fake 16x16 image
            from random noise vectors of shape (16,)
        - discriminator (keras.Model): Neural network classifi image real/fake,
            takes 16x16x1 images as input
    """

    # --------- GENERATOR ---------
    def get_generator():
        """
        Build the generator model for the GAN.

        The generator takes a noise vector of shape (16,) and progressively
        upsamples it through dense and convolutional layers
        to produce a 16x16x1 image.

        Returns:
            keras.Model: Generator model with input shape (16,)
            and output shape (16, 16, 1)
        """
        inputs = keras.Input(shape=(16,))                 # (None, 16)

        # Dense -> 2 * 2 * 512 = 2048   # (None, 2048)
        x = keras.layers.Dense(2048, activation="tanh")(inputs)

        # Reshape en petit "cube" 2x2x512      # (None, 2, 2, 512)
        x = keras.layers.Reshape((2, 2, 512))(x)

        # UpSampling 2x -> 4x4
        x = keras.layers.UpSampling2D()(x)              # (None, 4, 4, 512)
        x = keras.layers.Conv2D(64, (3, 3), padding="same")(x)  # (None,4,4,64)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("tanh")(x)

        # UpSampling 2x -> 8x8
        x = keras.layers.UpSampling2D()(x)                    # (None,8,8,64)
        x = keras.layers.Conv2D(16, (3, 3), padding="same")(x)  # (None,8,8,16)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("tanh")(x)

        # UpSampling 2x -> 16x16
        x = keras.layers.UpSampling2D()(x)                   # (None,16,16,16)
        x = keras.layers.Conv2D(1, (3, 3), padding="same")(x)  # (None,16,16,1)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Activation("tanh")(x)

        generator = keras.Model(inputs, outputs, name="generator")
        return generator

    # --------- DISCRIMINATOR ---------
    def get_discriminator():
        """
        Build the discriminator model for the GAN.

        The discriminator takes a 16x16x1 image and progressively downsample it
        through convolutional layers to produce a single output
        (real or fake classification).

        Returns:
            keras.Model: Discriminator model with input shape
            (16, 16, 1) and output shape (1,)
        """
        inputs = keras.Input(shape=(16, 16, 1))      # (None, 16,16,1)
        # (None,16,16,32)
        x = keras.layers.Conv2D(32, (3, 3), padding="same")(inputs)
        x = keras.layers.MaxPooling2D()(x)                # (None,8,8,32)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(64, (3, 3), padding="same")(x)  # (None,8,8,64)
        x = keras.layers.MaxPooling2D()(x)                      # (None,4,4,64)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(128, (3, 3), padding="same")(x)  # (N,4,4,128)
        x = keras.layers.MaxPooling2D()(x)                       # (N,2,2,128)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(256, (3, 3), padding="same")(x)  # (N,2,2,256)
        x = keras.layers.MaxPooling2D()(x)                       # (N,1,1,256)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Flatten()(x)                            # (None,256)
        outputs = keras.layers.Dense(1, activation="tanh")(x)    # (None,1)

        discriminator = keras.Model(inputs, outputs, name="discriminator")
        return discriminator

    return get_generator(), get_discriminator()
