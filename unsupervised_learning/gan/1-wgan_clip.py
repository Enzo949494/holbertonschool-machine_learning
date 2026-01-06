#!/usr/bin/env python3
"""
Wasserstein GAN with Gradient Clipping (WGAN-clip) implementation.

This module implements a Wasserstein GAN that uses weight clipping
to enforce the Lipschitz constraint on the discriminator.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """
    Wasserstein GAN with gradient clipping.

    Uses the Wasserstein distance (Earth-Mover distance) as the loss function
    and weight clipping to enforce Lipschitz constraints on the discriminator.
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """
        Initialize the WGAN with weight clipping.

        Args:
            generator: Keras model that generates fake samples
            discriminator: Keras model that classifies real vs fake samples
            latent_generator: Function that generates random latent vectors
            real_examples: Tensor of real training examples
            batch_size: Number of samples per batch (default: 200)
            disc_iter: Number of discriminator iterations per
                       generator iteration (default: 2)
            learning_rate: Learning rate for Adam optimizer (default: 0.005)
        """
        super().__init__()

        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # ----- WGAN losses -----

        # generator loss: - E[D(fake)]
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

        # discriminator loss: E[D(fake)] - E[D(real)]
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
            size: Number of samples to generate (default: batch_size)
            training: Whether to use training mode (default: False)

        Returns:
            Tensor of generated fake samples
        """
        if not size:
            size = self.batch_size
        latents = self.latent_generator(size)
        return self.generator(latents, training=training)

    def get_real_sample(self, size=None):
        """
        Get a random batch of real samples.

        Args:
            size: Number of samples to retrieve (default: batch_size)

        Returns:
            Tensor of real samples
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """
        Perform one training step.

        Trains the discriminator multiple times with weight clipping,
        then the generator once.

        Args:
            useless_argument: Unused argument (required by Keras API)

        Returns:
            Dictionary containing discriminator and generator losses
        """

        # 1) entraîner le discriminateur plusieurs fois
        for _ in range(self.disc_iter):
            real_sample = self.get_real_sample()
            fake_sample = self.get_fake_sample(training=True)

            with tf.GradientTape() as tape:
                d_real = self.discriminator(real_sample, training=True)
                d_fake = self.discriminator(fake_sample, training=True)
                discr_loss = self.discriminator.loss(d_fake, d_real)

            grads = tape.gradient(discr_loss,
                                  self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

            # weight clipping du discriminateur dans [-1, 1]
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        # 2) entraîner le générateur une fois
        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            d_fake = self.discriminator(fake_sample, training=True)
            gen_loss = self.generator.loss(d_fake)

        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        # 3) retourner les pertes pour fit()
        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
