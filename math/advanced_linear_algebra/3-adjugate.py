#!/usr/bin/env python3
"""Module for adjugate function"""
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """Calculate the adjugate matrix of a matrix"""
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or matrix == [[]] or any(
            len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    cofactor_matrix = cofactor(matrix)
    return [[cofactor_matrix[j][i] for j in range(len(cofactor_matrix))]
            for i in range(len(cofactor_matrix))]
