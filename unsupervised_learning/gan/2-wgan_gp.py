#!/usr/bin/env python3
"""
Wasserstein GAN with Gradient Penalty (WGAN-GP) implementation.

This module implements a WGAN-GP model that combines the Wasserstein distance
loss with gradient penalty to stabilize GAN training and prevent mode collapse.
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """
    Wasserstein GAN with Gradient Penalty.

    This class implements a WGAN with gradient penalty constraint to improve
    training stability. It trains a discriminator multiple times per generator
    update and applies a gradient penalty on interpolated samples.

    Attributes:
        generator: Keras model that generates fake samples
        discriminator: Keras model that discriminates real from fake samples
        latent_generator: Function that generates random latent vectors
        real_examples: Tensor of real training examples
        batch_size: Number of samples per batch
        disc_iter: Number of discriminator training steps per generator step
        learning_rate: Learning rate for Adam optimizer
        lambda_gp: Coefficient for gradient penalty loss
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        """
        Initialize WGAN-GP model.

        Args:
            generator: Keras model for generating samples
            discriminator: Keras model for discrimination
            latent_generator: Callable generates latent vectors of given size
            real_examples: Tensor of real training samples
            batch_size: Batch size for training (default: 200)
            disc_iter: Number of discriminator iterations per
                       generator update (default: 2)
            learning_rate: Learning rate for Adam optimizer (default: 0.005)
            lambda_gp: Gradient penalty coefficient (default: 10)
        """
        super().__init__()  # Keras.Model init

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

        # ----- losses WGAN (mêmes que WGAN_clip) -----

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

        # discriminator loss (sans penalty): E[D(fake)] - E[D(real)]
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
        Generate a batch of fake samples from the generator.

        Args:
            size: Number of samples to generate (default: batch_size)
            training: Whether the generator is in training mode (default:False)

        Returns:
            Tensor of generated fake samples
        """
        if not size:
            size = self.batch_size
        latents = self.latent_generator(size)
        return self.generator(latents, training=training)

    def get_real_sample(self, size=None):
        """
        Sample a batch of real examples from the dataset.

        Args:
            size: Number of samples to draw (default: batch_size)

        Returns:
            Tensor of real samples
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Generate interpolated samples between real and fake samples.

        Uses random weighted combinations: u*real + (1-u)*fake
        where u is drawn uniformly from [0, 1].

        Args:
            real_sample: Tensor of real samples
            fake_sample: Tensor of fake samples

        Returns:
            Tensor of interpolated samples
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Calculate gradient penalty on interpolated samples.

        The penalty encourages the gradient norm of the discriminator's output
        with respect to its input to be close to 1.

        Args:
            interpolated_sample: Tensor of interpolated samples

        Returns:
            Scalar tensor representing the gradient penalty loss
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """
        Execute one training step of the WGAN-GP model.

        Performs multiple discriminator updates followed by 1 generator update,
        with gradient penalty applied to the discriminator loss.

        Args:
            useless_argument: Unused argument (required by Keras Model API)

        Returns:
            Dictionary containing:
                - discr_loss: Discriminator loss without penalty
                - gen_loss: Generator loss
                - gp: Gradient penalty value
        """

        # 1) entraîner le discriminateur plusieurs fois
        for _ in range(self.disc_iter):
            real_sample = self.get_real_sample()
            fake_sample = self.get_fake_sample(training=True)
            interpolated_sample = self.get_interpolated_sample(
                real_sample, fake_sample
            )

            with tf.GradientTape() as tape:
                d_real = self.discriminator(real_sample, training=True)
                d_fake = self.discriminator(fake_sample, training=True)

                # loss WGAN "classique"
                discr_loss = self.discriminator.loss(d_fake, d_real)

                # gradient penalty
                gp = self.gradient_penalty(interpolated_sample)

                # nouvelle loss = loss WGAN + lambda * gp
                new_discr_loss = discr_loss + self.lambda_gp * gp

            grads = tape.gradient(
                new_discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

        # 2) entraîner le générateur une fois
        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            d_fake = self.discriminator(fake_sample, training=True)
            gen_loss = self.generator.loss(d_fake)

        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        # 3) retourner les pertes
        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
