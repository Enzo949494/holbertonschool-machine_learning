#!/usr/bin/env python3
"""Module for DeepNeuralNetwork class"""

import numpy as np


class DeepNeuralNetwork:
    """Class that defines a deep neural network
       performing binary classification"""

    def __init__(self, nx, layers):
        """
        Constructor for DeepNeuralNetwork class

        Args:
            nx: number of input features
            layers: list representing the number of nodes in each layer

        Raises:
            TypeError: if nx is not an integer
            ValueError: if nx is less than 1
            TypeError: if layers is not a list
            TypeError: if elements in layers are not positive integers
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Une seule boucle qui fait validation ET initialisation
        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                prev_layer_size = nx
            else:
                prev_layer_size = layers[i - 1]

            current_layer_size = layers[i]

            # Initialize weights using He et al. method
            self.__weights[f"W{i + 1}"] = np.random.randn(current_layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size)

            # Initialize biases to zeros
            self.__weights[f"b{i + 1}"] = np.zeros((current_layer_size, 1))

    @property
    def L(self):
        """Getter for number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
               nx is the number of input features to the neuron
               m is the number of examples

        Returns:
            The output of the neural network and the cache, respectively
        """
        # Save input to cache as A0
        self.__cache['A0'] = X

        # Forward propagation through all layers
        A = X
        for i in range(1, self.__L + 1):
            W = self.__weights[f'W{i}']
            b = self.__weights[f'b{i}']
            Z = np.dot(W, A) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache[f'A{i}'] = A

        return A, self.__cache

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
        Evaluates the neural network's predictions

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
        A, _ = self.forward_prop(X)

        # Convert probabilities to binary predictions
        # 1 if A >= 0.5, 0 otherwise
        predictions = np.where(A >= 0.5, 1, 0)

        # Calculate the cost
        cost = self.cost(Y, A)

        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
               for the input data
            cache: dictionary containing all the intermediary values of the network
            alpha: the learning rate

        Updates the private attribute __weights
        """
        m = Y.shape[1]

        A_L = cache[f'A{self.__L}']
        dZ = A_L - Y

        # Backward propagation through all layers using one loop
        for i in range(self.__L, 0, -1):
            A_prev = cache[f'A{i-1}']

            dW = (1/m) * np.dot(dZ, A_prev.T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

            # Save W before update for backpropagation
            if i > 1:
                W_current = self.__weights[f'W{i}'].copy()

            # Update weights and biases
            self.__weights[f'W{i}'] = self.__weights[f'W{i}'] - alpha * dW
            self.__weights[f'b{i}'] = self.__weights[f'b{i}'] - alpha * db

            # Calculate dZ for previous layer using weights before update
            if i > 1:
                A_prev = cache[f'A{i-1}']
                dZ = np.dot(W_current.T, dZ) * A_prev * (1 - A_prev)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the deep neural network

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
               nx is the number of input features to the neuron
               m is the number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
               for the input data
            iterations: number of iterations to train over
            alpha: the learning rate

        Raises:
            TypeError: if iterations is not an integer
            ValueError: if iterations is not positive
            TypeError: if alpha is not a float
            ValueError: if alpha is not positive

        Returns:
            The evaluation of the training data after iterations of training
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)