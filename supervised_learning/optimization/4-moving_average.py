#!/usr/bin/env python3
"""
Module for calculating weighted moving average
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set

    Args:
        data: list of data to calculate the moving average of
        beta: weight used for the moving average

    Returns:
        list containing the moving averages of data
    """
    moving_averages = []
    v = 0  # Initialize the moving average

    for i, value in enumerate(data):
        # Update the moving average: v = beta * v + (1 - beta) * current_value
        v = beta * v + (1 - beta) * value

        # Apply bias correction: divide by (1 - beta^(t+1))
        # where t is the current time step (starting from 0)
        bias_correction = 1 - beta ** (i + 1)
        corrected_average = v / bias_correction

        moving_averages.append(corrected_average)

    return moving_averages
