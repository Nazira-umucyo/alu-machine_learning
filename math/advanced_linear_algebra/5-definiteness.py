#!/usr/bin/env python3
"""Module for definiteness function"""
import numpy as np


def definiteness(matrix):
    """Calculate the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.size == 0 or len(matrix.shape) != 2:
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.array_equal(matrix, matrix.T):
        return None
    eigenvalues = np.linalg.eigvalsh(matrix)
    if all(e > 0 for e in eigenvalues):
        return "Positive definite"
    if all(e >= 0 for e in eigenvalues):
        return "Positive semi-definite"
    if all(e < 0 for e in eigenvalues):
        return "Negative definite"
    if all(e <= 0 for e in eigenvalues):
        return "Negative semi-definite"
    return "Indefinite"
