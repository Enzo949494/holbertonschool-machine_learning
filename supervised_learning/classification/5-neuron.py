#!/usr/bin/env python3
"""Module for Neuron class"""

import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Constructor for Neuron class

        Args:
            nx: number of input features to the neuron

        Raises:
            TypeError: if nx is not an integer
            ValueError: if nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Initialize private weights using random normal distribution
        self.__W = np.random.normal(size=(1, nx))
        # Initialize private bias to 0
        self.__b = 0
        # Initialize private activated output to 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
               nx is the number of input features to the neuron
               m is the number of examples

        Returns:
            The activated output of the neuron (self.__A)
        """
        # Calculate the linear transformation: z = W * X + b
        z = np.dot(self.__W, X) + self.__b

        # Apply sigmoid activation function: A = 1 / (1 + e^(-z))
        self.__A = 1 / (1 + np.exp(-z))

        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
               for the input data
            A: numpy.ndarray with shape (1, m) containing the activated output
               of the neuron for each example

        Returns:
            The cost
        """
        # Get number of examples
        m = Y.shape[1]

        # Calculate logistic regression cost
        # Cost = -1/m * sum(Y*log(A) + (1-Y)*log(1-A))
        # Using 1.0000001 - A instead of 1 - A to avoid division by zero
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
               nx is the number of input features to the neuron
               m is the number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
               for the input data

        Returns:
            The neuron's prediction and the cost of the network, respectively
            The prediction is a numpy.ndarray with shape (1, m) containing
            the predicted labels for each example
        """
        # Get the activated output using forward propagation
        A = self.forward_prop(X)

        # Convert probabilities to binary predictions
        # 1 if A >= 0.5, 0 otherwise
        predictions = np.where(A >= 0.5, 1, 0)

        # Calculate the cost
        cost = self.cost(Y, A)

        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
               nx is the number of input features to the neuron
               m is the number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
               for the input data
            A: numpy.ndarray with shape (1, m) containing the activated output
               of the neuron for each example
            alpha: the learning rate

        Updates the private attributes __W and __b
        """
        # Get number of examples
        m = Y.shape[1]

        # Calculate the gradient of the cost with respect to A
        dA = A - Y

        # Calculate the gradient of the cost with respect to W
        dW = (1/m) * np.dot(dA, X.T)

        # Calculate the gradient of the cost with respect to b
        db = (1/m) * np.sum(dA)

        # Update weights and bias using gradient descent
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db
