#!/usr/bin/env python3
"""Module that defines the Exponential distribution class"""


class Exponential:
    """Class that represents an Exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor

        Args:
            data: list of data to be used to estimate the distribution
            lambtha: expected number of occurrences in a given time frame
        """
        if data is None:
            # Use the given lambtha
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Calculate lambtha from data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # For Exponential distribution, lambtha = 1 / mean
            mean = sum(data) / len(data)
            self.lambtha = 1 / mean
