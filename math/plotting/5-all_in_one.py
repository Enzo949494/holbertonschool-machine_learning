#!/usr/bin/env python3
"""
figure with 5 plot
"""


import numpy as np
import matplotlib.pyplot as plt

def all_in_one():
    """
    figure with 5 plot
    """

    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # Figure
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('All in one')
    
    # Plot 1
    plt.subplot(3, 2, 1)
    plt.plot(y0, 'r-')
    plt.xlim(0, 10)
    plt.ylabel('y', fontsize='x-small')
    plt.yticks([0, 500, 1000])

    # Plot 2
    plt.subplot(3, 2, 2)
    plt.scatter(x1, y1, c='magenta', s=1)
    plt.xlabel('Height (in)', fontsize='x-small')
    plt.ylabel('Weight (lbs)', fontsize='x-small')
    plt.title("Men's Height vs Weight", fontsize='x-small')
    plt.xticks([60, 70, 80])
    plt.yticks([170, 180, 190])

    # Plot 3
    plt.subplot(3, 2, 3)
    plt.semilogy(x2, y2)
    plt.xlim(0, 28650)
    plt.xlabel('Time (years)', fontsize='x-small')
    plt.ylabel('Fraction Remaining', fontsize='x-small')
    plt.title('Exponential Decay of C-14', fontsize='x-small')
    plt.xticks([0, 10000, 20000])

    # Plot 4
    plt.subplot(3, 2, 4)
    plt.plot(x3, y31, 'r--', label='C-14')
    plt.plot(x3, y32, 'g-', label='Ra-226')
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.xlabel('Time (years)', fontsize='x-small')
    plt.ylabel('Fraction Remaining', fontsize='x-small')
    plt.title('Exponential Decay of Radioactive Elements', fontsize='x-small')
    plt.legend(fontsize='x-small')
    plt.xticks([0, 5000, 10000, 15000, 20000])
    plt.yticks([0.0, 0.5, 1.0])

    # Plot 5
    plt.subplot(3, 2, (5, 6))
    bins = range(0, 101, 10)
    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.xticks(range(0, 101, 10))
    plt.yticks([0, 10, 20, 30])
    plt.xlabel('Grades', fontsize='x-small')
    plt.ylabel('Number of Students', fontsize='x-small')
    plt.title('Project A', fontsize='x-small')

    plt.tight_layout()
    plt.show()
