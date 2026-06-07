#!/usr/bin/env python3
"""Module for poly_integral function"""


def poly_integral(poly, C=0):
    """Calculate the integral of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, int):
        return None
    result = [C]
    for i in range(len(poly)):
        coef = poly[i] / (i + 1)
        if coef == int(coef):
            coef = int(coef)
        result.append(coef)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result
