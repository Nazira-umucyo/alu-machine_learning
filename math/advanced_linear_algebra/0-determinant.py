#!/usr/bin/env python3
"""Module for determinant function"""


def determinant(matrix):
    """Calculate the determinant of a matrix"""
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for j in range(len(matrix)):
        sub = [[matrix[i][k] for k in range(len(matrix)) if k != j]
               for i in range(1, len(matrix))]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub)
    return det
