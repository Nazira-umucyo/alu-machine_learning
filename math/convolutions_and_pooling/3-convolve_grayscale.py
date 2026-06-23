#!/usr/bin/env python3
"""Module for strided convolution on grayscale images"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Perform a convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(np.ceil((kh - 1) / 2))
        pw = int(np.ceil((kw - 1) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    elif isinstance(padding, tuple):
        ph, pw = padding
    else:
        ph, pw = 0, 0
    images = images.astype('float64')
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, oh, ow))
    for i in range(oh):
        for j in range(ow):
            output[:, i, j] = np.sum(
                padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw] *
                kernel, axis=(1, 2))
    return output
