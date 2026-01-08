#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras


def convolutional_GenDiscr():

    # --------- GENERATOR ---------
    def get_generator():
        inputs = keras.Input(shape=(16,))                 # (None, 16)

        # Dense -> 2 * 2 * 512 = 2048
        x = keras.layers.Dense(2048, activation="tanh")(inputs)   # (None, 2048)

        # Reshape en petit "cube" 2x2x512
        x = keras.layers.Reshape((2, 2, 512))(x)                  # (None, 2, 2, 512)

        # UpSampling 2x -> 4x4
        x = keras.layers.UpSampling2D()(x)                        # (None, 4, 4, 512)
        x = keras.layers.Conv2D(64, (3, 3), padding="same")(x)    # (None, 4, 4, 64)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("tanh")(x)

        # UpSampling 2x -> 8x8
        x = keras.layers.UpSampling2D()(x)                        # (None, 8, 8, 64)
        x = keras.layers.Conv2D(16, (3, 3), padding="same")(x)    # (None, 8, 8, 16)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("tanh")(x)

        # UpSampling 2x -> 16x16
        x = keras.layers.UpSampling2D()(x)                        # (None, 16, 16, 16)
        x = keras.layers.Conv2D(1, (3, 3), padding="same")(x)     # (None, 16, 16, 1)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Activation("tanh")(x)

        generator = keras.Model(inputs, outputs, name="generator")
        return generator

    # --------- DISCRIMINATOR ---------
    def get_discriminator():
        inputs = keras.Input(shape=(16, 16, 1))                   # (None, 16,16,1)

        x = keras.layers.Conv2D(32, (3, 3), padding="same")(inputs)   # (None,16,16,32)
        x = keras.layers.MaxPooling2D()(x)                            # (None,8,8,32)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(64, (3, 3), padding="same")(x)        # (None,8,8,64)
        x = keras.layers.MaxPooling2D()(x)                            # (None,4,4,64)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(128, (3, 3), padding="same")(x)       # (None,4,4,128)
        x = keras.layers.MaxPooling2D()(x)                            # (None,2,2,128)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(256, (3, 3), padding="same")(x)       # (None,2,2,256)
        x = keras.layers.MaxPooling2D()(x)                            # (None,1,1,256)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Flatten()(x)                                 # (None,256)
        outputs = keras.layers.Dense(1, activation="tanh")(x)         # (None,1)

        discriminator = keras.Model(inputs, outputs, name="discriminator")
        return discriminator

    return get_generator(), get_discriminator()
