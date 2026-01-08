#!/usr/bin/env python3
"""
Module for Wasserstein GAN with Gradient Penalty (WGAN-GP).

This module implement a WGAN-GP model that trains a generator and discriminator
using the Wasserstein distance with gradient penalty to
improve training stability.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """
    Wasserstein GAN with Gradient Penalty (WGAN-GP).

    A GAN implementation that uses Wasserstein distance as the loss function
    and applies gradient penalty to enforce the
    Lipschitz constraint on the discriminator.

    Attributes:
        generator (keras.Model): Generator network that creates fake samples
        discriminator (keras.Model): Discriminator network classifies samples
        latent_generator (callable): Function to generate random latent vectors
        real_examples (tf.Tensor): Real training examples
        batch_size (int): Batch size for training
        disc_iter (int): Number of discriminator training
                         iterations per generator iteration
        learning_rate (float): Learning rate for optimizers
        lambda_gp (float): Weight for gradient penalty loss
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        """
        Initialize the WGAN-GP model.

        Args:
            generator (keras.Model): Generator neural network
            discriminator (keras.Model): Discriminator neural network
            latent_generator (callable): Function to generate latent
                                         vectors of shape (size, latent_dim)
            real_examples (tf.Tensor): Tensor of real training examples
            batch_size (int, optional): Batch size for training. Defaults 200.
            disc_iter (int, optional): Number of discriminator iterations
                                       per generator iteration. Defaults to 2.
            learning_rate (float, optional): Learning rate for Adam optimizers.
                                             Defaults to 0.005.
            lambda_gp (float, optional): Weight for gradient penalty loss.
                                         Defaults to 10.
        """
        super().__init__()

        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .3
        self.beta_2 = .9

        # ----- paramètres pour la gradient penalty -----
        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype="int32")
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # ----- losses WGAN (comme WGAN_clip) -----
        # generator: -E[D(fake)]
        self.generator.loss = lambda x: -tf.math.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss,
        )

        # discriminator: E[D(fake)] - E[D(real)]
        self.discriminator.loss = lambda x, y: (
            tf.math.reduce_mean(x) - tf.math.reduce_mean(y)
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss,
        )

    def get_fake_sample(self, size=None, training=False):
        """
        Generate fake samples using the generator.

        Args:
            size (int, optional): Number of samples to generate.
                                  Defaults to batch_size.
            training (bool, optional): Whether in training mode.
                                       Defaults to False.

        Returns:
            tf.Tensor: Tensor of generated fake samples
        """
        if not size:
            size = self.batch_size
        latents = self.latent_generator(size)
        return self.generator(latents, training=training)

    def get_real_sample(self, size=None):
        """
        Get a random batch of real samples from the training data.

        Args:
            size (int, optional): Number of samples to retrieve.
                                  Defaults to batch_size.

        Returns:
            tf.Tensor: Tensor of real samples randomly selected
                       from real_examples
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Create interpolated samples between real and fake samples.

        Used for computing the gradient penalty.
        Interpolates between real and fake
        samples using random weights.

        Args:
            real_sample (tf.Tensor): Batch of real samples
            fake_sample (tf.Tensor): Batch of fake samples

        Returns:
            tf.Tensor: Interpolated samples with same shape as inputs
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Compute the gradient penalty for the discriminator.

        The gradient penalty enforces that the discriminator's
        gradient has norm 1,
        which is required for the Lipschitz constraint.

        Args:
            interpolated_sample (tf.Tensor): Interpolated samples
                                             between real and fake

        Returns:
            tf.Tensor: Scalar gradient penalty loss
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def replace_weights(self, gen_h5, disc_h5):
        """
        Load pre-trained weights for generator and discriminator.

        Args:
            gen_h5 (str): Path to the generator weights file (.h5)
            disc_h5 (str): Path to the discriminator weights file (.h5)
        """
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)

    def train_step(self, useless_argument):
        """
        Perform one training step (forward and backward pass).

        Trains the discriminator multiple times (disc_iter) before
        training the generator once.
        This follows the WGAN-GP training procedure.

        Args:
            useless_argument: Not used, required by keras.Model interface

        Returns:
            dict: Dictionary containing:
                - 'discr_loss': Discriminator Wasserstein loss
                - 'gen_loss': Generator loss
                - 'gp': Gradient penalty value
        """

        # entraîner le discriminateur plusieurs fois
        for _ in range(self.disc_iter):
            real_sample = self.get_real_sample()
            fake_sample = self.get_fake_sample(training=True)
            interpolated_sample = self.get_interpolated_sample(
                real_sample, fake_sample
            )

            with tf.GradientTape() as tape:
                d_real = self.discriminator(real_sample, training=True)
                d_fake = self.discriminator(fake_sample, training=True)

                discr_loss = self.discriminator.loss(d_fake, d_real)
                gp = self.gradient_penalty(interpolated_sample)
                new_discr_loss = discr_loss + self.lambda_gp * gp

            grads = tape.gradient(
                new_discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

        # entraîner le générateur une fois
        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            d_fake = self.discriminator(fake_sample, training=True)
            gen_loss = self.generator.loss(d_fake)

        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
