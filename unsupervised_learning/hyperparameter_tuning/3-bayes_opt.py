#!/usr/bin/env python3
"""Module for Bayesian Optimization"""

import numpy as np
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