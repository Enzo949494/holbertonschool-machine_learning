#!/usr/bin/env python3
"""Module that defines the Poisson distribution class"""


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

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of "successes"

        Args:
            k: number of "successes"

        Returns:
            PMF value for k, or 0 if k is out of range
        """
        # Convert k to integer
        k = int(k)
        
        # k must be non-negative for Poisson distribution
        if k < 0:
            return 0
        
        # Calculate factorial of k
        factorial_k = 1
        for i in range(1, k + 1):
            factorial_k *= i
        
        # Calculate e^(-lambtha)
        e = 2.7182818285
        e_power = e ** (-self.lambtha)
        
        # PMF formula: (lambtha^k * e^(-lambtha)) / k!
        pmf_value = (self.lambtha ** k * e_power) / factorial_k
        
        return pmf_value
