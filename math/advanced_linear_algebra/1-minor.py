#!/usr/bin/env python3
"""Module for minor function"""
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """Calculate the minor matrix of a matrix"""
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or matrix == [[]] or any(
            len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]
    result = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix)):
            sub = [[matrix[r][c] for c in range(len(matrix)) if c != j]
                   for r in range(len(matrix)) if r != i]
            row.append(determinant(sub))
        result.append(row)
    return result
