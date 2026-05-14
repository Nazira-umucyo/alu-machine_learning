#!/usr/bin/env python3
"""Module for matrix_shape function"""


def matrix_shape(matrix):
    """Calculate the shape of a matrix"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
