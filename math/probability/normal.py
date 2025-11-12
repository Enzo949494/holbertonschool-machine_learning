#!/usr/bin/env python3
"""Module that defines the Normal distribution class"""


class Normal:
    """Class that represents a Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Class constructor

        Args:
            data: list of data to be used to estimate the distribution
            mean: mean of the distribution
            stddev: standard deviation of the distribution
        """
        if data is None:
            # Use the given mean and stddev
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            # Calculate mean and stddev from data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            
            # Calculate mean
            n = len(data)
            self.mean = sum(data) / n
            
            # Calculate standard deviation
            # stddev = sqrt(sum((x - mean)^2) / n)
            variance = sum((x - self.mean) ** 2 for x in data) / n
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value

        Args:
            x: x-value

        Returns:
            z-score of x
        """
        # z = (x - mean) / stddev
        z = (x - self.mean) / self.stddev
        return z

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score

        Args:
            z: z-score

        Returns:
            x-value of z
        """
        # x = mean + z * stddev
        x = self.mean + z * self.stddev
        return x