#!/usr/bin/env python3
"""Module for mean_cov function"""
import numpy as np


def mean_cov(X):
    """Calculate the mean and covariance of a data set"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0, keepdims=True)
    diff = X - mean
    cov = np.dot(diff.T, diff) / (n - 1)
    return mean, cov
