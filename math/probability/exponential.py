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

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period

        Args:
            x: time period

        Returns:
            PDF value for x, or 0 if x is out of range
        """
        # x must be non-negative for Exponential distribution
        if x < 0:
            return 0
        
        # Calculate e^(-lambtha * x)
        e = 2.7182818285
        e_power = e ** (-self.lambtha * x)
        
        # PDF formula: lambtha * e^(-lambtha * x)
        pdf_value = self.lambtha * e_power
        
        return pdf_value

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period

        Args:
            x: time period

        Returns:
            CDF value for x, or 0 if x is out of range
        """
        # x must be non-negative for Exponential distribution
        if x < 0:
            return 0
        
        # Calculate 1 - e^(-lambtha * x)
        e = 2.7182818285
        e_power = e ** (-self.lambtha * x)
        cdf_value = 1 - e_power
        
        return cdf_value
