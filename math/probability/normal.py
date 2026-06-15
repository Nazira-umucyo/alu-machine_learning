#!/usr/bin/env python3
"""Module for Normal distribution"""


class Normal:
    """Represents a Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize Normal distribution"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """Calculate z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculate x-value of a given z-score"""
        return self.mean + z * self.stddev

    def pdf(self, x):
        """Calculate PDF for a given x-value"""
        pi = 3.1415926536
        e = 2.7182818285
        coef = 1 / (self.stddev * (2 * pi) ** 0.5)
        exp = e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)
        return coef * exp

    def cdf(self, x):
        """Calculate CDF for a given x-value"""
        pi = 3.1415926536
        mean = self.mean
        stddev = self.stddev
        value = (x - mean) / (stddev * (2 ** 0.5))
        erf = (2 / (pi ** 0.5)) * (value - (value ** 3) / 3 +
                                    (value ** 5) / 10 -
                                    (value ** 7) / 42 +
                                    (value ** 9) / 216)
        return (1 + erf) / 2
