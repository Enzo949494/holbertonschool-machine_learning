#!/usr/bin/env python3
"""ResNet-50 architecture implementation"""

from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in Deep Residual
    Learning for Image Recognition (2015)

    Returns:
        the keras model
    """
    # He normal initializer with seed 0
    initializer = K.initializers.HeNormal(seed=0)

    # Input layer
    input_layer = K.Input(shape=(224, 224, 3))

    # Initial convolution
    conv1 = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same',
        kernel_initializer=initializer)(input_layer)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    # Max pooling
    pool1 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(act1)

    # Stage 2 (conv2_x)
    conv2 = projection_block(pool1, [64, 64, 256], s=1)
    conv2 = identity_block(conv2, [64, 64, 256])
    conv2 = identity_block(conv2, [64, 64, 256])

    # Stage 3 (conv3_x)
    conv3 = projection_block(conv2, [128, 128, 512], s=2)
    conv3 = identity_block(conv3, [128, 128, 512])
    conv3 = identity_block(conv3, [128, 128, 512])
    conv3 = identity_block(conv3, [128, 128, 512])

    # Stage 4 (conv4_x)
    conv4 = projection_block(conv3, [256, 256, 1024], s=2)
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])

    # Stage 5 (conv5_x)
    conv5 = projection_block(conv4, [512, 512, 2048], s=2)
    conv5 = identity_block(conv5, [512, 512, 2048])
    conv5 = identity_block(conv5, [512, 512, 2048])

    # Average pooling
    avg_pool = K.layers.AveragePooling2D(
        (7, 7), strides=(1, 1))(conv5)

    # Fully connected layer
    output = K.layers.Dense(
        1000, activation='softmax',
        kernel_initializer=initializer)(avg_pool)

    # Create model
    model = K.Model(inputs=input_layer, outputs=output)

    return model