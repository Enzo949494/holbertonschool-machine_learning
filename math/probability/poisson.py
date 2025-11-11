#!/usr/bin/env python3

class Poisson:
    """Class that represents a Poisson distribution"""

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
            # For Poisson distribution, lambtha is the mean of the data
            self.lambtha = float(sum(data) / len(data))
