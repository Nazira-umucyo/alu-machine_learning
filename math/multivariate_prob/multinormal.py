#!/usr/bin/env python3
"""Module for MultiNormal class"""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """Initialize MultiNormal distribution"""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        diff = data - self.mean
        self.cov = np.dot(diff, diff.T) / (n - 1)

    def pdf(self, x):
        """Calculate the PDF at a data point"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        diff = x - self.mean
        coef = 1 / (((2 * np.pi) ** d * det) ** 0.5)
        exp = np.exp(-0.5 * np.dot(np.dot(diff.T, inv), diff))
        return float(coef * exp)
