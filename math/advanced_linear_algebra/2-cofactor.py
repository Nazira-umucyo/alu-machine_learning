#!/usr/bin/env python3
"""Module for cofactor function"""
minor = __import__('1-minor').minor


def cofactor(matrix):
    """Calculate the cofactor matrix of a matrix"""
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or matrix == [[]] or any(
            len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    minor_matrix = minor(matrix)
    result = []
    for i in range(len(minor_matrix)):
        row = []
        for j in range(len(minor_matrix)):
            row.append(((-1) ** (i + j)) * minor_matrix[i][j])
        result.append(row)
    return result
