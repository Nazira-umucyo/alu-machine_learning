#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification."""

import numpy as np


class DeepNeuralNetwork:
    """Represents a deep neural network performing binary
    classification."""

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network.

        Args:
            nx (int): number of input features.
            layers (list): number of nodes in each layer.

        Raises:
            TypeError: if nx is not an integer.
            ValueError: if nx is less than 1.
            TypeError: if layers is not a list of positive integers.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev = nx
        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            self.__weights['W' + str(i + 1)] = (
                np.random.randn(layers[i], prev) * np.sqrt(2 / prev))
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
            prev = layers[i]

    @property
    def L(self):
        """Getter for the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for the cache dictionary."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights dictionary."""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): shape (nx, m), the input data.

        Returns:
            The output of the neural network and the cache, respectively.
        """
        self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            A_prev = self.__cache['A' + str(i - 1)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache['A' + str(i)] = A

        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): shape (1, m), correct labels.
            A (numpy.ndarray): shape (1, m), activated output.

        Returns:
            The cost.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.

        Args:
            X (numpy.ndarray): shape (nx, m), input data.
            Y (numpy.ndarray): shape (1, m), correct labels.

        Returns:
            A tuple (prediction, cost).
        """
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost
