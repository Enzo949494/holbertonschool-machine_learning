#!/usr/bin/env python3
"""
curve a red line y 
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    plot the curve y 0->10
    solid red line, x limited 0 to 10
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 11)
    plt.plot(x, y, 'r-')
    plt.xlim(0, 10)
    plt.show()
