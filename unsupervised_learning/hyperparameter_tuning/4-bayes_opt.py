#!/usr/bin/env python3
"""Module for Bayesian Optimization"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process"""
    
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Initialize the Bayesian Optimization
        
        Args:
            f: the black-box function to be optimized
            X_init: numpy.ndarray of shape (t, 1) - inputs already sampled
            Y_init: numpy.ndarray of shape (t, 1) - outputs for each input in X_init
            bounds: tuple of (min, max) representing the bounds of the space
            ac_samples: number of samples to analyze during acquisition
            l: length parameter for the kernel (default 1)
            sigma_f: standard deviation of the black-box function output (default 1)
            xsi: exploration-exploitation factor (default 0.01)
            minimize: bool for minimization vs maximization (default True)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.xsi = xsi
        self.minimize = minimize
        
        # Create evenly spaced acquisition sample points
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
    
    def acquisition(self):
        """
        Calculate the next best sample location using Expected Improvement
        
        Returns:
            X_next: numpy.ndarray of shape (1,) - next best sample point
            EI: numpy.ndarray of shape (ac_samples,) - expected improvement of each potential sample
        """
        # Get predictions from GP
        mu, sigma = self.gp.predict(self.X_s)
        
        # Find the best value seen so far
        if self.minimize:
            Y_best = np.min(self.gp.Y)
        else:
            Y_best = np.max(self.gp.Y)
        
        # Calculate improvement
        if self.minimize:
            improvement = Y_best - mu
        else:
            improvement = mu - Y_best
        
        # Calculate Expected Improvement
        # Avoid division by zero with a small epsilon
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = (improvement - self.xsi) / sigma
            EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0
        
        # Find the point with maximum EI
        X_next = self.X_s[np.argmax(EI)]
        
        return X_next, EI
