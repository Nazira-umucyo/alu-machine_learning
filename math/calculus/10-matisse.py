#!/usr/bin/env python3
"""Module for poly_derivative function"""


def poly_derivative(poly):
    """Calculate the derivative of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    result = [poly[i] * i for i in range(1, len(poly))]
    if result == []:
        return [0]
    return result
