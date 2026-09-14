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

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        prev = nx
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            self.weights['W' + str(i + 1)] = (
                np.random.randn(layers[i], prev) * np.sqrt(2 / prev))
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
            prev = layers[i]
