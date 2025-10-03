#!/usr/bin/env python3
"""
Module implementing the LeNet-5 architecture using TensorFlow/Keras.

Defines the lenet5 function to build and compile a LeNet-5 model.
"""

from tensorflow import keras as K


def lenet5(X):
    he_init = K.initializers.HeNormal(seed=0)

    # Couche 1 : conv 6 filtres 5x5, padding same, ReLU
    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                            activation='relu', kernel_initializer=he_init)(X)
    # MaxPooling 2x2, stride 2x2
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # Couche 2 : conv 16 filtres 5x5, padding valid, ReLU
    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                            activation='relu', kernel_initializer=he_init)(
                                pool1)
    # MaxPooling 2x2, stride 2x2
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # Flatten pour passer aux fully connected
    flat = K.layers.Flatten()(pool2)

    # Dense 120 neurones, ReLU
    fc1 = K.layers.Dense(
        120, activation='relu', kernel_initializer=he_init)(flat)
    # Dense 84 neurones, ReLU
    fc2 = K.layers.Dense(
        84, activation='relu', kernel_initializer=he_init)(fc1)
    # Sortie 10 neurones, softmax
    out = K.layers.Dense(
        10, activation='softmax', kernel_initializer=he_init)(fc2)

    model = K.Model(inputs=X, outputs=out)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model
