#!/usr/bin/env python3
"""Module for Gaussian Process regression"""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process"""
    
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initialize the Gaussian Process
        
        Args:
            X_init: numpy.ndarray of shape (t, 1) - input samples
            Y_init: numpy.ndarray of shape (t, 1) - output samples
            l: length parameter for the kernel (default 1)
            sigma_f: standard deviation of the black-box function (default 1)
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)
    
    def kernel(self, X1, X2):
        """
        Calculate the covariance kernel matrix using RBF kernel
        
        Args:
            X1: numpy.ndarray of shape (m, 1)
            X2: numpy.ndarray of shape (n, 1)
        
        Returns:
            numpy.ndarray of shape (m, n) - covariance kernel matrix
        """
        # RBF kernel: sigma_f^2 * exp(-||x1 - x2||^2 / (2 * l^2))
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-sqdist / (2 * self.l**2))
    
    def predict(self, X_s):
        """
        Predict the mean and standard deviation at sample points
        
        Args:
            X_s: numpy.ndarray of shape (s, 1) - sample points
        
        Returns:
            mu: numpy.ndarray of shape (s,) - mean at each sample point
            sigma: numpy.ndarray of shape (s,) - variance at each sample point
        """
        # Kernel between training data and sample points
        K_s = self.kernel(self.X, X_s)
        
        # Kernel between sample points and themselves
        K_ss = self.kernel(X_s, X_s)
        
        # Inverse of training kernel matrix
        K_inv = np.linalg.inv(self.K)
        
        # Mean prediction
        mu = K_s.T.dot(K_inv).dot(self.Y).flatten()
        
        # Variance prediction
        sigma = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s)).flatten()
        
        return mu, sigma
    
    def update(self, X_new, Y_new):
        """
        Update the Gaussian Process with a new sample point
        
        Args:
            X_new: numpy.ndarray of shape (1,) - new sample point
            Y_new: numpy.ndarray of shape (1,) - new sample function value
        
        Updates X, Y, and K attributes
        """
        # Reshape X_new and Y_new to (1, 1)
        X_new = X_new.reshape(1, 1)
        Y_new = Y_new.reshape(1, 1)
        
        # Append new sample to existing data
        self.X = np.vstack([self.X, X_new])
        self.Y = np.vstack([self.Y, Y_new])
        
        # Recompute the kernel matrix
        self.K = self.kernel(self.X, self.X)