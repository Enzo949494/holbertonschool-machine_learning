#!/usr/bin/env python3
"""
plot bar
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    plot bar
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    fruits = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    x = np.arange(len(people))
    width = 0.5

    bottom = np.zeros(len(people))

    for i in range(len(fruits)):
        plt.bar(x, fruit[i], width, bottom=bottom, color=colors[i],
                label=fruits[i])
        bottom += fruit[i]

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.xticks(x, people)
    plt.yticks(range(0, 81, 10))
    plt.ylim(0, 80)
    plt.legend()

    plt.show()
