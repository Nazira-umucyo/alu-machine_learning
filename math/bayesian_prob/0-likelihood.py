#!/usr/bin/env python3
"""Module for likelihood function"""
import numpy as np


def likelihood(x, n, P):
    """Calculate the likelihood of obtaining data given hypothetical probs"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    factorial_n = 1
    for i in range(1, n + 1):
        factorial_n *= i
    factorial_x = 1
    for i in range(1, x + 1):
        factorial_x *= i
    factorial_nx = 1
    for i in range(1, n - x + 1):
        factorial_nx *= i
    comb = factorial_n / (factorial_x * factorial_nx)
    return comb * (P ** x) * ((1 - P) ** (n - x))
