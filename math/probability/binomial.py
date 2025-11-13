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

        # k must be between 0 and n for Binomial distribution
        if k < 0 or k > self.n:
            return 0

        # Calculate binomial coefficient C(n, k) = n! / (k! * (n-k)!)
        # Using the formula: C(n, k) = n * (n-1) * ... * (n-k+1) / k!
        binomial_coef = 1
        for i in range(k):
            binomial_coef = binomial_coef * (self.n - i) / (i + 1)

        # PMF formula: C(n, k) * p^k * (1-p)^(n-k)
        pmf_value = binomial_coef * (self.p ** k) * ((1 - self.p) **
                                                     (self.n - k))

        return pmf_value
