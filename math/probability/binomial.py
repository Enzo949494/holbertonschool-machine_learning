#!/usr/bin/env python3
"""Module that defines the Binomial distribution class"""


class Binomial:
    """Class that represents a Binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Class constructor

        Args:
            data: list of data to be used to estimate the distribution
            n: number of Bernoulli trials
            p: probability of a "success"
        """
        if data is None:
            # Use the given n and p
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            # Calculate n and p from data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean and variance
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # For binomial distribution:
            # mean = n * p
            # variance = n * p * (1 - p)
            # From variance: variance = mean * (1 - p)
            # So: (1 - p) = variance / mean
            # Therefore: p = 1 - (variance / mean)
            p = 1 - (variance / mean)

            # Calculate n from mean and p
            # n = mean / p
            n = mean / p

            # Round n to nearest integer
            self.n = round(n)

            # Recalculate p with the rounded n
            self.p = mean / self.n
